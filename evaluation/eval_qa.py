"""Code for test huatuo"""

import os
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
from accelerate import Accelerator, DeepSpeedPlugin
import transformers
from transformers import set_seed, get_cosine_schedule_with_warmup
import datasets
import shutil
import json
import random
from scorer import score_mix2


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelWithLMHead
# from models.tokenization_moss import MossTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path,tokenizer, few_shot_num):
        self.config = config
        self.query_prompt = "请回答下面选择题。\n[{question_type}]{question}\n{option_str}"
        self.query_prompt_without_ty = "请回答下面选择题。\n{question}\n{option_str}"

        self.dataset = {}
        with open(data_path) as f:
            self.dataset = json.load(f)
        self.datas = []

        self.tokenizer = tokenizer
        self.user_token = self.tokenizer.convert_ids_to_tokens(195)
        self.assistant_token = self.tokenizer.convert_ids_to_tokens(196)

        for sou, sou_dataset in self.dataset.items():
            for da in sou_dataset:
                if 'option' in da:
                    da['query'] = self.get_query(da)
                else:
                    da['query'] = self.generate_prompt(da['query'],None)
                da['dataset'] = sou
                self.datas.append(da)
        # debug
        # self.datas = self.datas[:100]

    def get_query(self, da):
        da['option_str'] = '\n'.join([f'{k}. {v}' for k,v in da['option'].items() if len(v) > 0])
        if 'question_type' in da:
            da['query'] = self.query_prompt.format_map(da)
        else:
            da['query'] = self.query_prompt_without_ty.format_map(da)
        da['query'] = self.generate_prompt(da['query'],None)
        return da['query']

    def generate_prompt(self,query, history):
        if 'HuatuoGPT-II' in self.config.model_path or 'huatuo' in self.config.model_path.lower() :
            if not history:
                return  f"<问>：{query}\n<答>："
                # return  f"""一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。<病人>：{query} <HuatuoGPT>："""
            else:
                prompt = ''
                for i, (old_query, response) in enumerate(history):
                    prompt += "<问>：{}\n<答>：{}\n".format(old_query, response)
                prompt += "<问>：{}\n<答>：".format(query)
                return prompt
            
        elif 'Baichuan2-7B-Chat' in self.config.model_path or 'Baichuan2-13B-Chat' in self.config.model_path:
            prompt = ''
            if history:
                for i, (old_query, response) in enumerate(history):
                    prompt += f"{self.user_token} {old_query} {self.assistant_token} {response}"
            prompt += f"{self.user_token} {query} {self.assistant_token}"
            return prompt
        
        elif 'DISC-MedLLM' in self.config.model_path:
            prompt = ''
            self.user_token = '<reserved_102>'
            self.assistant_token = '<reserved_103>'
            if history:
                for i, (old_query, response) in enumerate(history):
                    prompt += f"{self.user_token} {old_query} {self.assistant_token} {response}"
            prompt += f"{self.user_token} {query} {self.assistant_token}"
            return prompt
        
        elif 'PMC_LLaMA_13B' in self.config.model_path:
            return query

        elif 'zhongjing' in self.config.model_path:
            return self.tokenizer.bos_token + ' <human> ' + query.strip() + '\n<bot>'

        elif 'chatglm' in self.config.model_path or 'Qwen-' in self.config.model_path:
            return query
        
        elif 'bianque' in self.config.model_path:
            return "病人：" + query + "\n医生："
        
        elif 'llama2-' in self.config.model_path:
            prompt = ''
            if history:
                for i, (old_query, response) in enumerate(history):
                    prompt += f"[INST] {old_query} [/INST] {response} </s><s>"
            prompt += f"[INST] {query} [/INST]"
            return prompt
        
        elif 'huatuogpt_7b' in self.config.model_path:
            # 第一代huatuogpt
            if not history:
                return f"""一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。<病人>：{query} <HuatuoGPT>："""
            else:
                prompt = '一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。'
                for i, (old_query, response) in enumerate(history):
                    prompt += ("<病人>：{} <HuatuoGPT>：{}".format(old_query, response))
                prompt += "<病人>：{} <HuatuoGPT>：".format(query)
                return prompt

    def __getitem__(self, index):
        da = self.datas[index]
        return {
            'data': da,
            'input': da['query']
        }
    
    def __len__(self):
        return len(self.datas)
    
    def collate_fn(self, batch):
        batch_query = [x['input'] for x in batch]
        batch_data = [x['data'] for x in batch]
        out_batch = {}
        out_batch['data'] = batch_data

        output_tokenizer = self.tokenizer(batch_query, return_tensors='pt', padding='longest')
        out_batch['input_ids'] = output_tokenizer['input_ids']
        out_batch['attention_mask'] = output_tokenizer['attention_mask']
        max_length = 1024
        if output_tokenizer['input_ids'].shape[-1] > max_length:
            out_batch['input_ids'] = out_batch['input_ids'][:,-max_length:]
            out_batch['attention_mask'] = out_batch['attention_mask'][:,-max_length:]

        # dubug
        # print(out_batch['input_ids'].shape,out_batch['attention_mask'].shape)
        # print(out_batch['input_ids'][0],out_batch['attention_mask'][0], self.tokenizer.decode(out_batch['input_ids'][0]))

        return out_batch

def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
    # for output in outputs:
    # responses = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]
    batch_return=[]
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i%num_return==num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def table_to_csv_string(table):
    rows = [",".join(table.columns)]  
    for row in table.data:
        rows.append(",".join(map(str, row)))
    return "\n".join(rows)

def test(args):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    accelerator.print(f'args:\n{args}')



    if 'tfmr' in  args.model_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    elif 'chatglm' in args.model_path or 'bianque' in args.model_path:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half()
    elif 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
        model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.cuda().eval()
    accelerator.print(f'load_finish')

    args.batch_size = 2
    args.max_new_tokens = 768
    if 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
        left_tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
        args.batch_size = 1
        args.max_new_tokens = 128
    else:
        left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

    if 'Qwen-' in args.model_path:
        left_tokenizer.pad_token_id = 151643
        left_tokenizer.eos_token_id = 151643

    if left_tokenizer.pad_token is None:
        # left_tokenizer.pad_token = '<PAD>'
        left_tokenizer.pad_token = '</s>' 

    dataset = TestDataset(args, args.data_path , left_tokenizer, '')
    # 注意batch_size
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    args.num_return = 1
    # gen_kwargs = {'num_return_sequences': args.num_return, 'max_new_tokens': args.max_new_tokens, 'num_beams': 1,
    #      'do_sample':True, 'top_p':0.7, 'temperature': 1, 'repetition_penalty':1.1}

    gen_kwargs = {'num_return_sequences': args.num_return, 'max_new_tokens': args.max_new_tokens, 'num_beams': 1,
         'do_sample':True, 'top_p':0.7, 'temperature':0.5, 'repetition_penalty':1.1}
    
    # gen_kwargs = {'num_return_sequences': args.num_return, 'max_new_tokens': args.max_new_tokens, 'num_beams': 1,
    #      'do_sample':False, 'top_p':0.7, 'temperature':0.5, 'repetition_penalty':1.1}

    val_dataloader = accelerator.prepare( val_dataloader)
    accelerator.wait_for_everyone()
    cache_data = []
    cache_response = []

    # accelerator.print(accelerator.deepspeed_config)
    with torch.no_grad():
        ress = []

        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader

        for batch in dataloader_iterator:
            input_ids = batch["input_ids"]
            data = batch["data"]
            attention_mask = batch["attention_mask"]
            if 'chatglm' in args.model_path or 'Qwen-' in args.model_path or 'bianque' in args.model_path:
                response = []
                for da in batch["data"]:
                    res, history = model.chat(left_tokenizer, da['query'], history=None)
                    response.append([res])
            elif 'PMC_LLaMA_13B' in args.model_path:
                outputs = model.generate(input_ids)
                response = get_response(input_ids,outputs,left_tokenizer,args.num_return)
            else:
                outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                response = get_response(input_ids,outputs,left_tokenizer,args.num_return)
            
            cache_data.extend(data)
            cache_response.extend(response)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        all_data =  [None] * dist.get_world_size()
        all_response =  [None] * dist.get_world_size()

        dist.all_gather_object(all_response,cache_response)
        dist.all_gather_object(all_data,cache_data)

        all_data = [item for sublist in all_data for item in sublist]
        all_response = [item for sublist in all_response for item in sublist]

        for d, r in zip(all_data,all_response):
            for ind,_r in enumerate(r):
                d[f'huatuo_answer_{ind}'] = _r
            ress.append(d)

        if accelerator.is_main_process:
            task_name = os.path.split(args.model_path)[-1].split(".")[0] if 'tfmr' not in args.model_path else '-'.join(args.model_path.split('/')[-3:-1])
            task_name =  task_name + f'_{os.path.split(args.data_path)[-1].replace(".json","")}'
            run_time = 0
            for i in range(100):
                run_time += 1
                out_file = f'{task_name}_t{run_time}.json'
                if not os.path.exists(out_file):
                    break

            print(f'test results: {out_file}')
            val_res = score_mix2(out_file,ress,True,1)
            val_res_table = table_to_csv_string(val_res['InputOutputTable'])
            del val_res['InputOutputTable']
            outstr = json.dumps(val_res,ensure_ascii=False,indent = 2)
            accelerator.print(outstr)
            outstr += '\n' + json.dumps(val_res_table,ensure_ascii=False,indent = 2)
            outstr += '\n'+f'output: {out_file}'
            with open(f'result/{task_name}_t{run_time}.json','w', encoding='utf-8') as f:
                f.write(outstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--data_path', default='.data/eval_qa.json', type=str)
    parser.add_argument('--model_path', default='FreedomIntelligence/HuatuoGPT2-7B', type=str)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    test(args)           
