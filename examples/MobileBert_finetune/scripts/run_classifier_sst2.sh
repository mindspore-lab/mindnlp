CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/mobilebert
export DATA_DIR=$CURRENT_DIR/dataset/glue_data
export OUTPUR_DIR=$CURRENT_DIR/outputs
export CUDA_VISIBLE_DEVICES=5
TASK_NAME="SST-2"

python run_classifier.py \
  --model_type=mobilebert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=5.0 \
  --max_grad_norm=1.0 \
  --logging_steps=2105 \
  --save_steps=2105 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42\

