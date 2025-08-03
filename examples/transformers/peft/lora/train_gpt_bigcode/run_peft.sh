export HF_ENDPOINT=https://hf-mirror.com

python train.py \
    --model_path "bigcode/gpt_bigcode-santacoder" \
    --dataset_name "smangrul/hf-stack-v1" \
    --subset "data" \
    --data_column "content" \
    --split "train" \
    --seq_length 512 \
    --max_steps 2000 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --num_warmup_steps 30 \
    --num_workers 4 \
    --bf16 \
    --no_fp16 \
    --output_dir "peft-lora-santaCoder" \
    --fim_rate 0.5 \
    --fim_spm_rate 0.5 \
    --use_peft_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --use_4bit_qunatization \
    --use_nested_quant \
    --bnb_4bit_compute_dtype "bfloat16" \
    --num_epochs 10
