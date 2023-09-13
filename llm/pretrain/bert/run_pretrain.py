import time
import argparse
import numpy as np
import os
import sys
import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ms_function
from mindspore.ops import cross_entropy, clip_by_global_norm
from mindspore.amp import all_finite
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from src.bert import BertForPretraining
from src.api import value_and_grad
from src.optimization import create_optimizer
from src.config import BertConfig, PretrainedConfig
from src.utils import get_output_file_time, save_bert_small_checkpoint

# get pwd 
def getpwd():
    pwd = sys.path[0]
    if os.path.isfile(pwd):
        pwd = os.path.dirname(pwd)
    return pwd

def train(model, optimizer, train_dataset, train_batch_size, jit=True, print_iters=100):
    """
    Train function for Bert pre-training.
    """
    # 5. Define forward and grad function.
    def forward_fn(input_ids, input_mask, segment_ids, \
                   masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label):
        (prediction_scores, seq_relationship_score, _) = model(input_ids, input_mask, segment_ids, None, None, masked_lm_positions)

        masked_lm_loss = cross_entropy(prediction_scores.view(-1, prediction_scores.shape[-1]),
                                       masked_lm_ids.view(-1), masked_lm_weights.view(-1))
        next_sentence_loss = cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss

        return total_loss, masked_lm_loss, next_sentence_loss

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                   next_sentence_label, segment_ids):
        (total_loss, masked_lm_loss, next_sentence_loss), grads = grad_fn(input_ids, input_mask, segment_ids, \
                              masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
        grads = clip_by_global_norm(grads, clip_norm=1.0)
        grads = grad_reducer(grads)
        status = all_finite(grads)
        status = ops.AllReduce()(status)
        if status:
            total_loss = ops.depend(total_loss, optimizer(grads))
        return total_loss, masked_lm_loss, next_sentence_loss, status

    if jit:
        train_step = ms_function(train_step)

    # 6. train
    total = train_dataset.get_dataset_size()
    print(f"Train Begin, batch size: {train_batch_size}, total training steps: {total}")
    # train begin
    loss_total, masked_lm_loss_total, next_sentence_loss_toal = 0, 0, 0
    cur_step_nums, train_step_nums, skip_step_nums = 0, 0, 0
    cur_time, avg_time = time.time(), 0
    # step begin
    model.set_train()
    for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
            next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
        total_loss, masked_lm_loss, next_sentence_loss, status = \
            train_step(input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                       next_sentence_label, segment_ids)
        status = status.asnumpy()
        if status:
            loss_total = loss_total + total_loss.asnumpy()
            masked_lm_loss_total = masked_lm_loss_total + masked_lm_loss.asnumpy()
            next_sentence_loss_toal = next_sentence_loss_toal + next_sentence_loss.asnumpy()
        else:
            print(f"grads overflow, skip step {cur_step_nums}")
            skip_step_nums += 1
        
        if train_step_nums % print_iters == 0 and train_step_nums != 0:
            print_time = time.time()
            total_time = print_time - cur_time
            cur_time = print_time
            avg_time = total_time / (print_iters + skip_step_nums)

            print(f"avg_time(ms): {avg_time * 1000:2f}, "
                f"rank_id: {rank_id}, "
                f"cur_step: {cur_step_nums}, "
                f"skip_steps: {skip_step_nums:3d}, "
                f"train_step: {train_step_nums}, "
                f"loss: {loss_total/cur_step_nums:f}, "
                f"masked_lm_loss: {masked_lm_loss_total/cur_step_nums:f}, "
                f"next_sentence_loss: {next_sentence_loss_toal/cur_step_nums:f}")

        # saving ckpt per 10000 steps or last step
        if args.do_save_ckpt:
            if (train_step_nums % args.save_steps == 0 or cur_step_nums == total - 1) and cur_step_nums != 0:
                print(f"saving ckpt on cur_step: {cur_step_nums}, train_step: {train_step_nums}, in card: {rank_id}")
                save_bert_small_checkpoint(cur_step_nums=train_step_nums,
                                         save_checkpoint_path=args.save_ckpt_path,
                                         rank_num=rank_id,
                                         network=model)
        if status:
            train_step_nums += 1
        cur_step_nums += 1
    print("Pretrain done!")


def init_args():
    # use time mark ouput file
    output_file_time = get_output_file_time()
    output_file = os.path.join(getpwd(), 'outputs' ,'model_save_' + output_file_time)
    if not os.path.exists(output_file):
        try:
            os.mkdir(output_file)
        except FileExistsError:
            pass
    # the follow are pretrain basic setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=config.dataset_mindreocrd_dir, type=str,\
                        help="Using data path for mindrecord.")
    parser.add_argument("--use_ascend", default='False', type=str, required=True,\
                        help="if use ascend mode.")               
    parser.add_argument("--jit", default=config.jit, type=bool,\
                        help="Choose jit mode.")
    parser.add_argument("--do_train", default=config.do_train, type=bool,\
                        help="Whether to run training.")
    parser.add_argument("--lr", default=config.lr, type=float, required=True,\
                        help="Learning rate for training.")
    parser.add_argument("--warmup_steps", default=config.warmup_steps, type=int, required=True,\
                        help="Warm up steps.")
    parser.add_argument("--train_batch_size", default=config.train_batch_size, type=int, required=True,\
                        help="Choose train batch size.")
    parser.add_argument("--epochs", default=config.epochs, type=int, required=True,\
                        help="Choose training epochs value.")
    parser.add_argument("--do_save_ckpt", default=config.do_save_ckpt, type=str,\
                        help="Whether to save ckpt.")
    parser.add_argument("--save_steps", default=config.save_steps, type=int,\
                        help="How many steps need to save ckpt.")
    parser.add_argument("--save_ckpt_path", default=output_file, type=str,\
                        help="Ckpt save path.")
    parser.add_argument("--do_load_ckpt", default=None, required=True, type=str,\
                        help="Whether need to load ckpt.")
    parser.add_argument("--model_path", default=None, type=str,\
                        help="Ckpt path to load.")
    parser.add_argument("--config", default=None, type=str, required=True,\
                        help="config path to load.")

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    args = parser.parse_args()
    if not os.path.exists(args.save_ckpt_path):
        os.mkdir(args.save_ckpt_path)
    return args

if __name__ == '__main__':
    config = PretrainedConfig()
    args = init_args()
    config = BertConfig(args.config)
    if args.use_ascend == "True":
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", device_id=rank_id)
    else: 
        init("nccl")
        rank_id = get_rank()
        rank_size = get_group_size()
        mindspore.set_context(enable_graph_kernel=True)

    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
                                        device_num = rank_size,
                                        gradients_mean = True)
    
    # 1. Read pre-train dataset.
    # dataset_path = args.data_dir
    # new dataset path is merge_and_save_mindrecord
    dataset_path = ['merge_and_save_mindrecord/bert_pretrain_data.mindrecord{index}'.format(index=i) for i in range(8)]
    train_dataset = ds.MindDataset(dataset_files=dataset_path, num_shards=rank_size, shard_id=rank_id)
    # 2. Batchify the dataset.
    train_dataset = train_dataset.batch(args.train_batch_size, drop_remainder=True)
    
    train_dataset = train_dataset.repeat(args.epochs)
    # 3. Define model.
    model = BertForPretraining(config)
    # load ckpt
    if args.do_load_ckpt == 'True':
        if args.model_path is not None:
            ckpt_file = args.model_path
            bert_dict = mindspore.load_checkpoint(ckpt_file)
            mindspore.load_param_into_net(model, bert_dict)
            print("load ckpt successful")
        else:
            raise ValueError("Need to input checkout file")
    # 4. Define optimizer(trick: warm up).
    num_train_steps = train_dataset.get_dataset_size()
    print(num_train_steps)
    optimizer = create_optimizer(model, args.lr, num_train_steps, args.warmup_steps)

    # 6. Pretrain
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    train(model, optimizer, train_dataset, args.train_batch_size, jit=args.jit)
