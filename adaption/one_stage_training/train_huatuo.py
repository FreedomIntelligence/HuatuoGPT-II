"""Code for finetune_huatuo"""

import os
os.environ["WANDB_API_KEY"]='Your wandb key'
import copy
import json
import torch
import logging
import argparse
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
import transformers
from transformers import set_seed, get_cosine_schedule_with_warmup
import datasets
import shutil
import json
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class HuatuoGPT2_train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = datasets.load_from_disk(config.data_path)
        self.debug = True

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        if self.debug:
            print(self.tokenizer.decode(batch[0]['input_ids']))
            self.debug = False

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }
        
    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
    

def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) 

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir)
    
    accelerator.print(f'args:\n{args}')

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps

    left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    

    if left_tokenizer.pad_token is None:
        left_tokenizer.pad_token = '<PAD>'

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = HuatuoGPT2_train_dataset(args, left_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * (args.n_epochs + 0.35)) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    if args.checkpoint_path:
        if os.path.isfile(os.path.join(args.checkpoint_path, "scheduler.bin")) and \
           os.path.isfile(os.path.join(args.checkpoint_path, "training_state.pt")):
            accelerator.load_state(args.checkpoint_path)
            training_state = torch.load(os.path.join(args.checkpoint_path, "training_state.pt"))
            start_epoch = training_state["epoch"]
            start_step = training_state["step"]+1
            global_step = training_state["global_step"]
            accelerator.print(f"Checkpoint Loaded at {start_epoch} epoch, {start_step} step and {global_step} global step")
            accelerator.print(f"Loading trained model :{args.checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint not found at: {args.checkpoint_path}")
    else:
        start_epoch = 0
        start_step = 0
        global_step = 0

    if args.save_step <= 0:
        args.save_step=len(train_dataloader) // 5
        accelerator.print(f'Save step setted to {args.save_step}')

    metric = SFTMetric(device=torch.cuda.current_device())

    def save_checkpoint(epoch, step, global_step):
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
        accelerator.wait_for_everyone()
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state(save_dir)
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    accelerator.print(accelerator.deepspeed_config)
    model.train()

    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']

            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

            if global_step % args.save_step == 22:
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
            
        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)
        start_step = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_path',default=None, type=str)
    parser.add_argument('--model_path', default='', type=str)
    
    # Data Args
    parser.add_argument('--not_shuffle_train_loader', action='store_true')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    
    # Training Args
    parser.add_argument('--max_seq_len', default=4096, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=1, type=int)

    # Other Args
    parser.add_argument('--save_step', default=-1, type=int)
    parser.add_argument('--eval_step', default=-1, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)