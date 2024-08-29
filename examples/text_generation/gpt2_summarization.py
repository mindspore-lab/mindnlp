import mindspore
import argparse
import numpy as np
import logging
import os

import json

from tqdm import tqdm
import random
from datetime import datetime
from mindnlp.core import nn, ops
from mindnlp.core.nn import CrossEntropyLoss
from mindnlp.transformers import GPT2Config, GPT2LMHeadModel, BertTokenizer
from mindspore import Tensor, Parameter, ops
from mindspore.train.serialization import save_checkpoint
from os.path import join


PAD = '[PAD]'
pad_id = 0
logger = None

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='GPT2-Summary-mindspore/config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='GPT2-Summary-mindspore/vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='GPT2-Summary-mindspore/data/train_with_summ.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='GPT2-Summary-mindspore/data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='GPT2-Summary-mindspore/data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', default=True, help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=6, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=5, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=2, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='GPT2-Summary-mindspore/summary_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='GPT2-Summary-mindspore/tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--train_mmi', default=False, help="若指定该参数，则训练DialoGPT的MMI模型")
    parser.add_argument('--train_mmi_tokenized_path', default='GPT2-Summary-mindspore/data/train_mmi_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
    parser.add_argument('--mmi_model_output_path', default='GPT2-Summary-mindspore/mmi_model', type=str, required=False, help='MMI模型保存路径')
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    mindspore.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(args, vocab_size):
    """
    :param args:
    :param vocab_size:字典大小
    :return:
    """
    if args.pretrained_model:  # 如果指定了预训练的GPT2模型
        model = GPT2LMHeadModel.load(args.pretrained_model)
        # model = gpt2.GPT2LMHeadModel.load(args.pretrained_model)
    else:  # 若没有指定预训练模型，则初始化模型
        # model_config = config_gpt2.GPT2Config.from_json_file(args.model_config)
        model_config =GPT2Config.from_json(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    # model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")


def preprocess_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(args.train_raw_path,
                                                                                    args.train_tokenized_path))

    with open(args.train_tokenized_path,"w",encoding="utf-8") as f:
        with open(args.train_raw_path, 'r',encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                try:
                    file_line = json.loads(line)
                except:
                    print ("line",line)

                else:
                    
                    # 另一种实现方式，这种方式会将每一句原始语料进行对齐，长度为n_ctx
                    # text1 = file_line['article']
                    # tokenizer_text1 = tokenizer(text1)
                    # tokenizer_text1 = tokenizer_text1.tolist()
                    # text2 = file_line['summarization']
                    # tokenizer_text2 = tokenizer(text2)
                    # tokenizer_text2 = tokenizer_text2.tolist()[1:][:-1]
                    # dialogue_ids = tokenizer_text1 + tokenizer_text2
                    # n_legth = n_ctx - 1
                    # if len(dialogue_ids) < n_legth:
                    #     # Pad the list with pad_id
                    #     padding = [pad_id] * (n_legth - len(dialogue_ids))
                    #     dialogue_ids.extend(padding)
                    # elif len(dialogue_ids) > n_legth:
                    #     # Truncate the list
                    #     dialogue_ids = dialogue_ids[:n_legth]
                    # dialogue_ids.append(sep_token_id)
                    
                    dialogue_ids = [cls_token_id]
                    dialogue_ids.extend([vocab.tokens_to_ids(word) 
                                         if vocab.tokens_to_ids(word) != -1 
                                         else unk_token_id 
                                         for word in file_line['article']])
                    dialogue_ids.append(sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    dialogue_ids.extend([vocab.tokens_to_ids(word) 
                                         if vocab.tokens_to_ids(word) != -1 
                                         else unk_token_id 
                                         for word in file_line['summarization']])
                    dialogue_ids.append(sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                    dialogue_ids = dialogue_ids[:n_ctx]
                    for dialogue_id in dialogue_ids:
                        f.write(str(dialogue_id) + ' ')
                    f.write("\n")
                    
                    
def preprocess_mmi_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料的每段对话进行翻转，然后转换为用于train MMI模型的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance N[SEP]utterance N-1[SEP]utterance N-2[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing MMI raw data,raw data path:{}, token output path:{}".format(args.train_raw_path,
                                                                                        args.train_mmi_tokenized_path))
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
    with open(args.train_mmi_tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in reversed(utterances):  # 将一段对话进行翻转
                dialogue_ids.extend([vocab.tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            # 最后一条记录不添加换行符
            if dialogue_index < len(train_data) - 1:
                f.write("\n")
    logger.info("finish preprocessing raw data,the result is stored in {}".format(args.train_tokenized_path))


def calculate_loss_and_accuracy(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:m  
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    shift_labels = shift_labels.astype(mindspore.int32)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1))
    
    preds = shift_logits.argmax(axis=-1) # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    # not_ignore = shift_labels != pad_id
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.astype(mindspore.int64).sum()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


@mindspore.jit_class
class Accumulator():
    """_summary_ 梯度更新类
    """
    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads):
        # 将单步获得的梯度累加至Accumulator的inner_grads
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            # 如果达到累积步数，进行参数优化更新
            self.optimizer(self.inner_grads)
            # 完成参数优化更新后，清零inner_grads
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        # 计算步数加一
        ops.assign_add(self.counter, Tensor(1, mindspore.int32))

        return True

args = setup_train_args()
logger = create_logger(args)

with open(args.vocab_path, "r", encoding="utf-8") as f:
    vocab_list = f.read().strip().split("\n")
# 初始化tokenizer
vocab = text.Vocab.from_list(vocab_list)
# mindspore中Windows平台尚不支持 BertTokenizer 。
tokenizer = BertTokenizer(vocab=vocab, lower_case=True, return_token=False)

cls_token_id = vocab.tokens_to_ids('[CLS]')
sep_token_id = vocab.tokens_to_ids('[SEP]')
unk_token_id = vocab.tokens_to_ids('[UNK]')

# tokenizer的字典大小
vocab_size = len(vocab_list)
# 加载GPT2模型
model, n_ctx = create_model(args, vocab_size)

accumulate_step = args.gradient_accumulation
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.AdamW(model.trainable_params(), learning_rate=args.lr)
accumulator = Accumulator(optimizer, accumulate_step)


# Define forward function
def forward_fn(data, label):
    """_summary_ 前向推理步骤

    Args:
        data (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    logits = model(data)
    loss, accuracy = calculate_loss_and_accuracy(logits, label)
    return loss / accumulate_step


# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())


# Define function of one-step training
# @mindspore.jit
def train_step(data, label):
    """_summary_ 训练步骤

    Args:
        data (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss, grads = grad_fn(data, label)
    loss = ops.depend(loss, accumulator(grads))
    return loss


def ckpt_to_mindspore(mth_file, size:str=None):
    """_summary_ 
    Resolve the parameter wte.embedding of the generative model,
    the lack of training transformer prefix in table allows for loading

    Args:
        mth_file (_type_): _description_
        size (str, optional): _description_. Defaults to None.

    Raises:
        ImportError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    try:
        import mindspore
    except:
        raise ImportError(f"'import mindspore' failed, please install mindspore by "
                          f"`pip mindspore torch` or instructions from 'https://www.mindspore.cn/install'")

    size = "mindspore" if not size else size # rename ckpt

    from mindspore import Tensor
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = mindspore.load_checkpoint(mth_file)

    for k, v in state_dict.items():
        if 'wte.embedding_table' in k:
            k = k.replace('wte.embedding_table', 'transformer.wte.embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    try:
        save_checkpoint(ms_ckpt, mth_file)
    except:
        raise RuntimeError(f'Save checkpoint to {mth_file} failed, please checkout the path.')

    return mth_file


def train(model, train_list, multi_gpu, args):
    """_summary_ 训练逻辑，进行数据加载之后的推理和保存模型

    Args:
        model (_type_): _description_
        train_list (_type_): _description_
        multi_gpu (_type_): _description_
        args (_type_): _description_

    Raises:
        exception: _description_
    """
    train_dataset = MyDataset(train_list)
    train_dataloader = ds.GeneratorDataset(train_dataset, column_names="input_ids" ,num_parallel_workers=args.num_workers, shuffle=True)
    train_dataloader = train_dataloader.padded_batch(batch_size=args.batch_size, drop_remainder=True, pad_info={})
    train_dataloader = train_dataloader.repeat(1)
    train_dataloader = train_dataloader.shuffle(10)
    
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    logger.info('starting training')
    # 记录 out of memory的次数
    oom_time = 0
    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        size = len(train_dataloader)
        # batch_idx为int，input_ids为一个tensor
        for batch_idx, input_ids in enumerate(train_dataloader.create_tuple_iterator()):
            # 注意：GPT2模型的construct()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                input_id = input_ids[0].astype(mindspore.int64)
                loss = train_step(input_id, input_id)
                
                if batch_idx % args.log_step == 0:
                    loss, current = loss.asnumpy(), batch_idx
                    logger.info(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))    
                else:
                    logger.info(str(exception))
                    raise exception
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if args.train_mmi:  # 当前训练MMI模型
            model_path = join(args.mmi_model_output_path, 'model_epoch{}'.format(epoch + 1))
        else:  # 当前训练对话模型
            model_path = join(args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_path = f"{model_path}/mindspore_model.ckpt"
        save_checkpoint(model_to_save, model_path)
        ckpt_to_mindspore(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('training finished')


def main():
    """_summary_ 主函数
    """
    # 日志同时输出到文件和console
    global logger
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    global pad_id
    # pad_id = tokenizer.convert_tokens_to_ids(PAD)
    pad_id = vocab.tokens_to_ids(PAD)

    # 创建对话模型的输出目录
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)
    # 创建MMI模型的输出目录
    if not os.path.exists(args.mmi_model_output_path):
        os.mkdir(args.mmi_model_output_path)

    # 对原始数据进行预处理,将原始语料转换成对应的token_id
    if args.raw and args.train_mmi:  # 如果当前是要训练MMI模型
        preprocess_mmi_raw_data(args, tokenizer, n_ctx)
    elif args.raw and not args.train_mmi:  # 如果当前是要训练对话生成模型
        print ("_______________________________________")
        # preprocess_raw_data(args, tokenizer, n_ctx)
        
    # 是否使用多块GPU进行并行运算
    multi_gpu = False
    
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.trainable_params()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 加载数据
    logger.info("loading traing data")
    if args.train_mmi:  # 如果是训练MMI模型
        with open(args.train_mmi_tokenized_path, "r", encoding="utf8") as f:
            data = f.read()
    else:  # 如果是训练对话生成模型
        with open(args.train_tokenized_path, "r", encoding="utf8") as f:
            data = f.read()
    data_list = [line.rstrip() for line in data.split("\n") if line.rstrip()]
    train_list = data_list
    
    # 开始训练
    train(model, train_list, multi_gpu, args)
    
    
if __name__ == '__main__':
    main()
    