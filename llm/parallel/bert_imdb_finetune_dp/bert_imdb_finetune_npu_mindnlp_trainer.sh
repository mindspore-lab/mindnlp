#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash bert_imdb_finetune_npu_mindnlp_trainer.sh"
echo "==========================================="

EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/data" ]; then
    if [ ! -f "${EXEC_PATH}/emotion_detection.tar.gz" ]; then
        wget https://baidu-nlp.bj.bcebos.com/emotion_detection-dataset-1.0.0.tar.gz -O emotion_detection.tar.gz
    fi
    tar xvf emotion_detection.tar.gz
fi
export DATA_PATH=${EXEC_PATH}/data/

rm -rf bert_imdb_finetune_cpu_mindnlp_trainer_npus_same
mkdir bert_imdb_finetune_cpu_mindnlp_trainer_npus_same
echo "start training"

export MULTI_NPU="true" 
export ASCEND_SLOG_PRINT_TO_STDOUT=1

msrun --worker_num=2 --local_worker_num=2 --master_port=8121 \
--log_dir=bert_imdb_finetune_cpu_mindnlp_trainer_npus_same --join=True \
--cluster_time_out=10 bert_imdb_finetune_cpu_mindnlp_trainer_npus_same.py 
