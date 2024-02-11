# InstructUIE

- This repo releases our implementation for the InstructUIE model.
- It is built based on the pretrained Flan T5 model, and finetuned on our data.

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.26.1)
- DeepSpeed (0.7.7)

You can install the required libraries by running 

```
bash setup.sh
```


## Data

Our models are trained and evaluated on **IE INSTRUCTIONS**. 
You can download the data from [Baidu NetDisk](https://pan.baidu.com/s/1R0KqeyjPHrsGcPqsbsh1XA?from=init&pwd=ybkt) or [Google Drive](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view?usp=share_link).


## Training

A sample script for training the InstructUIE model in our paper can be found at [`scripts/train_flan-t5.sh`](scripts/train_flan-t5.sh). You can run it as follows:

```
nohup bash ./scripts/train_flan-t5-base.sh >> result_train_flan-t5-base.txt 2>&1 &
```

```
nohup bash ./scripts/train_flan-t5-xxl.sh >> result_train_flan-t5-xxl.txt 2>&1 &
```

## Released Checkpoints

We have released our 11B UIE model, click [here](https://huggingface.co/ZWK/InstructUIE) for download.


## Evaluation

A sample script for evaluating the InstructUIE model in our paper can be found at [`scripts/eval_flan-t5.sh`](scripts/eval_flan-t5.sh). You can run it as follows:

```
nohup bash ./scripts/test_flan-t5-base.sh >> result_test_flan-t5-base.txt 2>&1 &
```

```
nohup bash ./scripts/test_flan-t5-xxl.sh >> result_test_flan-t5-xxl.txt 2>&1 &
```

The decoded results would save to predict_eval_predictions.jsonl in your output dir. 
To calculate f1 with predict_eval_predictions.jsonl

```
nohup python -u src/calculate_f1.py >> result_f1_flan-t5-base.txt 2>&1 &
```

```
nohup python src/calculate_f1.py >> result_f1_flan-t5-xxl.txt 2>&1 &
```

## Citation
```latex
@article{wang2023instructuie,
  title={InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction},
  author={Wang, Xiao and Zhou, Weikang and Zu, Can and Xia, Han and Chen, Tianze and Zhang, Yuansen and Zheng, Rui and Ye, Junjie and Zhang, Qi and Gui, Tao and others},
  journal={arXiv preprint arXiv:2304.08085},
  year={2023}
}
```


```
CUDA_VISIBLE_DEVICES=0 python src/run_uie.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/luohaoran/huggingface/google/flan-t5-base --data_dir IE_INSTRUCTIONS --task_config_dir configs/multi_task_configs --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir output/flan-t5-base-ie-single --input_record_file flan-t5-base_train.record --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 8 --learning_rate 5e-05 --num_train_epochs 10 --deepspeed configs/ds_configs/stage0.config --run_name flan-t5-base-mult-mi-experiment --max_source_length 512 --max_target_length 50 --generation_max_length 50 --max_num_instances_per_task 10000 --max_num_instances_per_eval_task 200 --add_task_name False --add_dataset_name False --num_examples 0 --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 500 --evaluation_strategy no --save_strategy steps --save_steps 2000
```
```
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client src/run_uie.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/luohaoran/huggingface/google/flan-t5-base --data_dir IE_INSTRUCTIONS --task_config_dir configs/multi_task_configs --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir output/flan-t5-base-ie-single --input_record_file flan-t5-base_train.record --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 8 --learning_rate 5e-05 --num_train_epochs 10 --deepspeed configs/ds_configs/stage0.config --run_name flan-t5-base-mult-mi-experiment --max_source_length 512 --max_target_length 50 --generation_max_length 50 --max_num_instances_per_task 10000 --max_num_instances_per_eval_task 200 --add_task_name False --add_dataset_name False --num_examples 0 --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 500 --evaluation_strategy no --save_strategy steps --save_steps 2000
```
```
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client src/data_process.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/luohaoran/huggingface/google/flan-t5-base --data_dir IE_INSTRUCTIONS --task_config_dir configs/multi_task_configs --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir output/flan-t5-base-ie-single --input_record_file flan-t5-base_train.record --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 8 --learning_rate 5e-05 --num_train_epochs 10 --deepspeed configs/ds_configs/stage0.config --run_name flan-t5-base-mult-mi-experiment --max_source_length 512 --max_target_length 50 --generation_max_length 50 --max_num_instances_per_task 10000 --max_num_instances_per_eval_task 200 --add_task_name False --add_dataset_name False --num_examples 0 --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 500 --evaluation_strategy no --save_strategy steps --save_steps 2000
```


```
CUDA_VISIBLE_DEVICES=0 python src/run_uie.py --do_predict --predict_with_generate --model_name_or_path /home/luohaoran/huggingface/google/flan-t5-base --resume_from_checkpoint output/flan-t5-base-ie-single --data_dir IE_INSTRUCTIONS --task_config_dir configs/multi_task_configs --instruction_file configs/instruction_config.json --instruction_strategy single --input_record_file flan-t5-base_test.record --per_device_eval_batch_size 16 --deepspeed configs/ds_configs/stage0.config --run_name flan-t5-base-mult-mi-experiment --max_source_length 512 --max_target_length 50 --generation_max_length 50 --max_num_instances_per_eval_task 200 --add_task_name False --add_dataset_name False --num_examples 0 --overwrite_output_dir --overwrite_cache --output_dir eval_output/flan-t5-base-ie-single
```
```
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client src/run_uie.py --do_predict --predict_with_generate --model_name_or_path /home/luohaoran/huggingface/google/flan-t5-base --resume_from_checkpoint output/flan-t5-base-ie-single --data_dir IE_INSTRUCTIONS --task_config_dir configs/multi_task_configs --instruction_file configs/instruction_config.json --instruction_strategy single --input_record_file flan-t5-base_test.record --per_device_eval_batch_size 16 --deepspeed configs/ds_configs/stage0.config --run_name flan-t5-base-mult-mi-experiment --max_source_length 512 --max_target_length 50 --generation_max_length 50 --max_num_instances_per_eval_task 200 --add_task_name False --add_dataset_name False --num_examples 0 --overwrite_output_dir --overwrite_cache --output_dir eval_output/flan-t5-base-ie-single
```








## SFT

```
CUDA_VISIBLE_DEVICES=4 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --dataset verbal_sft_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/verbal_sft/Llama-2-13B/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 4096 >> result_verbal_sft_train_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py --deepspeed ds_config.json --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --dataset verbal_sft_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/verbal_sft/Llama-2-13B/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 3072 >> result_verbal_sft_train_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=5 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --dataset verbal_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/verbal_sft_new2/Llama-2-13B/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new2_train_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

   

```
CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft/Llama-2-13B/checkpoint --dataset verbal_sft_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft/Llama-2-13B/evaluation --per_device_eval_batch_size 12 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_test_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=4 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft_new/Llama-2-13B/checkpoint --dataset verbal_sft_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft_new/Llama-2-13B/evaluation --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new_test_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=5 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft_new2/Llama-2-13B/checkpoint --dataset verbal_sft_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft_new2/Llama-2-13B/evaluation --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new2_test_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```





```
CUDA_VISIBLE_DEVICES=1 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft/Llama-2-13B/checkpoint --dataset verbal_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft/Llama-2-13B/evaluation_small --per_device_eval_batch_size 12 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_test_Llama-2-13B_qlora4_epoch5_small.txt 2>&1 &
```


```
CUDA_VISIBLE_DEVICES=2 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft_new/Llama-2-13B/checkpoint --dataset verbal_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft_new/Llama-2-13B/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new_test_Llama-2-13B_qlora4_epoch5_small.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=3 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft_new2/Llama-2-13B/checkpoint --dataset verbal_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft_new2/Llama-2-13B/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new2_test_Llama-2-13B_qlora4_epoch5_small.txt 2>&1 &
```












```
CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-Python-hf --dataset code_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/code_sft_new2/CodeLlama-13B-Python/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 4096 >> result_code_sft_new2_train_CodeLlama-2-13B-Python_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=1 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-hf --dataset code_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/code_sft_new2/CodeLlama-13B/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 4096 >> result_code_sft_new2_train_CodeLlama-2-13B_qlora4_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=4 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-Python-hf --dataset code_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/code_sft_new2/CodeLlama-13B-Python/qlora8/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 8 --max_length 4096 >> result_code_sft_new2_train_CodeLlama-2-13B-Python_qlora8_epoch5.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=5 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-hf --dataset code_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/code_sft_new2/CodeLlama-13B/qlora8/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 8 --max_length 4096 >> result_code_sft_new2_train_CodeLlama-2-13B_qlora8_epoch5.txt 2>&1 &
```




```
CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-Python-hf --adapter_name_or_path expr/code_sft_new2/CodeLlama-13B-Python/checkpoint --dataset code_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/code_sft_new2/CodeLlama-13B-Python/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 4096 >> result_code_sft_new2_test_CodeLlama-13B-Python_qlora4_epoch5_small.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=1 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-hf --adapter_name_or_path expr/code_sft_new2/CodeLlama-13B/checkpoint --dataset code_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/code_sft_new2/CodeLlama-13B/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 4096 >> result_code_sft_new2_test_CodeLlama-13B_qlora4_epoch5_small.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=4 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-Python-hf --adapter_name_or_path expr/code_sft_new2/CodeLlama-13B-Python/qlora8/checkpoint --dataset code_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/code_sft_new2/CodeLlama-13B-Python/qlora8/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 8 --max_length 4096 >> result_code_sft_new2_test_CodeLlama-13B-Python_qlora8_epoch5_small.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=5 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/codellama/CodeLlama-13b-hf --adapter_name_or_path expr/code_sft_new2/CodeLlama-13B/qlora8/checkpoint --dataset code_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/code_sft_new2/CodeLlama-13B/qlora8/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 8 --max_length 4096 >> result_code_sft_new2_test_CodeLlama-13B_qlora8_epoch5_small.txt 2>&1 &
```