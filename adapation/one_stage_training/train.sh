
data_file=./test_samples.json
model_dir=baichuan-inc/Baichuan2-13B-Base
experiment_name=HuatuoGPT2_7B

accelerate launch --config_file ./training_config/zero.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_huatuo.py \
    --experiment_name ${experiment_name}\
    --model_path ${model_dir}\
    --max_seq_len 4096 \
    --gradient_accumulation_steps 4 \
    --data_dir {} \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 2 \
    --warmup_rates 0.01 \
    --train_bsz_per_gpu 4 \
    --eval_bsz_per_gpu 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing > training.log 2>&1 &