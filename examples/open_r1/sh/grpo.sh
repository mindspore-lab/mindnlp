PYTHONPATH=/home/ma-user/work/mind-openr1/src python src/mind_openr1/grpo.py \
    --model_name_or_path /home/ma-user/work/Qwen2.5-1.5B \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config math \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --bf16 True \
    --torch_dtype bfloat16 \
    --output_dir checkpoints/Qwen2.5-1.5B-GRPO \
    --save_steps 100000

    # --dataset_name /home/ma-user/work/mind-openr1/data/open-r1___mixture-of-thoughts

# nohup bash sh/grpo.sh > /home/ma-user/work/mind-openr1/logs/grpo.log 2>&1 &

