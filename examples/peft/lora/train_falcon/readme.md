# 数据集下载
通过mrpc_dataset.py脚本中load_examples接口自动从Hugging Face下载并加载数据集。
# lora训练命令 
训练之前先确保已经下载好模型文件，文件默认存放目录为.mindnlp/model/Rocketknight1/falcon-rw-1b
在mindnlp根目录下执行如下命令
python llm/peft/train_falcon/train_mrpc.py \
    --save_dir ".mindnlp/peft_model/falcon/mrpc_lora" \
    --batch_size 8 \
    --model_name_or_path ".mindnlp/model/Rocketknight1/falcon-rw-1b" \
    --max_seq_len 256 \
    --lora