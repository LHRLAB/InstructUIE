# ChatUIE

## Data

Our models are trained and evaluated on **IE INSTRUCTIONS**. 
You can download the data from [Baidu NetDisk](https://pan.baidu.com/s/1R0KqeyjPHrsGcPqsbsh1XA?from=init&pwd=ybkt) or [Google Drive](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view?usp=share_link).


## SFT

### Train
```
CUDA_VISIBLE_DEVICES=5 nohup python -u src/train_bash.py --stage sft --do_train --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --dataset verbal_sft_new_train --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir expr/verbal_sft_new2/Llama-2-13B/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 5.0 --plot_loss --fp16 --overwrite_output_dir --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new2_train_Llama-2-13B_qlora4_epoch5.txt 2>&1 &
```

### Predict
```
CUDA_VISIBLE_DEVICES=3 nohup python -u src/train_bash.py --stage sft --do_predict --model_name_or_path /home/luohaoran/huggingface/meta-llama/Llama-2-13b-hf --adapter_name_or_path expr/verbal_sft_new2/Llama-2-13B/checkpoint --dataset verbal_sft_new_test --template llama2 --finetuning_type lora --output_dir expr/verbal_sft_new2/Llama-2-13B/evaluation_small --per_device_eval_batch_size 8 --predict_with_generate --fp16 --quantization_bit 4 --max_length 3072 >> result_verbal_sft_new2_test_Llama-2-13B_qlora4_epoch5_small.txt 2>&1 &
```
```
python src/calculate_f1_new.py
```
