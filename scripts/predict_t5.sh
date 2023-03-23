#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# TODO 将路径变为静态变量

# 3090 * 8 on t5-700M
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_uie.py \
   --do_predict \
   --predict_with_generate \
   --resume_from_checkpoint /workspace/InstructUIE/output/t5-700M-ie-single \
   --model_name_or_path /root/MODELS/flan-t5-700M \
   --output_dir /workspace/InstructUIE/output/t5-700M-ie-single \
   --data_dir /workspace/IE_data_v2 \
   --task_config_dir /workspace/InstructUIE/configs/multi_task_configs \
   --instruction_file /workspace/InstructUIE/configs/instruction_config.json \
   --instruction_strategy single \
   --per_device_eval_batch_size 16 \
   --deepspeed configs/ds_configs/eval.config \
   --max_source_length 512 \
   --max_target_length 50 \
   --max_predict_samples 20 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --logging_strategy steps \
   --logging_steps 50
