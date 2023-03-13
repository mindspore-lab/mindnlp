#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.dataset.text import Vocab

from mindnlp import load_dataset, process
from mindnlp.transforms import BasicTokenizer, PadTransform, Lookup, AddToken, Truncate


# In[2]:


split = ('train', 'valid', 'test')
language_pair = ('de', 'en')
train_dataset, valid_dataset, test_dataset = load_dataset('multi30k', split=split, language_pair=language_pair)


# In[3]:


tokenizer = BasicTokenizer(True)


# In[4]:


train_dataset = train_dataset.map(tokenizer, 'en').map(tokenizer, 'de')
valid_dataset = valid_dataset.map(tokenizer, 'en').map(tokenizer, 'de')
test_dataset = test_dataset.map(tokenizer, 'en').map(tokenizer, 'de')


# In[5]:


de_vocab = Vocab.from_dataset(train_dataset, 'de', freq_range=(2, None), \
                              special_tokens=['<bos>', '<eos>', '<pad>', '<unk>'], special_first=True)
en_vocab = Vocab.from_dataset(train_dataset, 'en', freq_range=(2, None), \
                              special_tokens=['<bos>', '<eos>', '<pad>', '<unk>'], special_first=True)


# In[6]:


max_len = 32

en_begin_add = AddToken('<bos>')
en_end_add = AddToken('<eos>', False)
en_lookup_op = Lookup(en_vocab, unknown_token='<unk>')
en_truncate = Truncate(max_len - 2)
en_pad_value = en_vocab.tokens_to_ids('<pad>')
en_pad_op = PadTransform(max_len, en_pad_value, return_length=True)

de_begin_add = AddToken('<bos>')
de_end_add = AddToken('<eos>', False)
de_lookup_op = Lookup(de_vocab, unknown_token='<unk>')
de_truncate = Truncate(max_len - 2)
de_pad_value = de_vocab.tokens_to_ids('<pad>')
de_pad_op = PadTransform(max_len, de_pad_value)


# In[7]:


train_dataset = train_dataset.map([en_begin_add, en_end_add, en_lookup_op, en_truncate, en_pad_op], \
                                  'en', ['en', 'en_len']) \
                            .map([de_begin_add, de_end_add, de_lookup_op, de_truncate, de_pad_op], 'de')

valid_dataset = valid_dataset.map([en_begin_add, en_end_add, en_lookup_op, en_truncate, en_pad_op], \
                                  'en', ['en', 'en_len']) \
                            .map([de_begin_add, de_end_add, de_lookup_op, de_truncate, de_pad_op], 'de')

test_dataset = test_dataset.map([en_begin_add, en_end_add, en_lookup_op, en_truncate, en_pad_op], \
                                  'en', ['en', 'en_len']) \
                            .map([de_begin_add, de_end_add, de_lookup_op, de_truncate, de_pad_op], 'de')


# In[8]:


train_dataset = train_dataset.batch(128, drop_remainder=True)
valid_dataset = valid_dataset.batch(128)
test_dataset = test_dataset.batch(128)


# In[9]:


class Encoder(nn.Cell):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # Embedding层
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)  # 双向GRU层
        self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim)  # 全连接层

        self.dropout = nn.Dropout(p=dropout)  # dropout，防止过拟合

    def construct(self, src, src_len):
        """构建编码器

        Args:
            src: 源序列，为已经转换为数字索引并统一长度的序列；shape = [src len, batch_size]
            src_len: 有效长度；shape = [batch_size, ]
        """

        # 将输入源序列转化为向量，并进行暂退（dropout）
        # shape = [src len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))
        # 计算输出
        # shape = [src len, batch size, enc hid dim*2]
        outputs, hidden = self.rnn(embedded, seq_length=src_len)
        # 为适配解码器，合并两个上下文函数
        # shape = [batch size, dec hid dim]
        hidden = ops.tanh(self.fc(ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)))

        return outputs, hidden


# In[10]:


class Attention(nn.Cell):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        # attention线性层
        self.attn = nn.Dense((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        # v， 用不带有bias的线性层表示
        # shape = [1, dec hid dim]
        self.v = nn.Dense(dec_hid_dim, 1, has_bias=False)

    def construct(self, hidden, encoder_outputs, mask):
        """Attention层

        Args:
            hidden: 解码器上一个时刻的隐藏状态；shape = [batch size, dec hid dim]
            encoder_outputs: 编码器的输出，前向与反向RNN的隐藏状态；shape = [src len, batch size, enc hid dim * 2]
            mask: 将<pad>占位符的注意力权重替换为0或者很小的数值；shape = [batch size, src len]
        """

        src_len = encoder_outputs.shape[0]

        # 重复解码器隐藏状态src len次，对齐维度
        # shape = [batch size, src len, dec hid dim]
        hidden = ops.tile(hidden.expand_dims(1), (1, src_len, 1))

        # 将编码器输出中的第1、2维度进行交换，对齐维度
        # shape = [batch size, src len, enc hid dim*2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # 计算E_t
        # shape = [batch size, src len, dec hid dim]
        energy = ops.tanh(self.attn(ops.concat((hidden, encoder_outputs), axis=2)))

        # 计算v * E_t
        # shape = [batch size, src len]
        attention = self.v(energy).squeeze(2)

        # 不需要考虑序列中<pad>占位符的注意力权重
        attention = attention.masked_fill(mask == 0, -1e10)

        return ops.softmax(attention, axis=1)


# In[11]:


class Decoder(nn.Cell):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, inputs, hidden, encoder_outputs, mask):
        """构建解码器

        Args:
            input: 输入的单词；shape = [batch size]
            hidden: 解码器上一时刻的隐藏状态；shape = [batch size, dec hid dim]
            encoder_outputs: 编码器的输出，前向与反向RNN的隐藏状态；shape = [src len, batch size, enc hid dim * 2]
            mask: 将<pad>占位符的注意力权重替换为0或者很小的数值；shape = [batch size, src len]
        """

        # 为输入增加额外维度
        # shape = [1, batch size]
        inputs = inputs.expand_dims(0)

        # 输入词的embedding输出， d(y_t)
        # shape = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(inputs))

        # 注意力权重向量, a_t
        # shape = [batch size, src len]
        a = self.attention(hidden, encoder_outputs, mask)

        # 为注意力权重增加额外维度
        # shape = [batch size, 1, src len]
        a = a.expand_dims(1)

        # 将编码器隐藏状态中的第1、2维度进行交换
        # shape = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # 计算w_t
        # shape = [batch size, 1, enc hid dim * 2]
        weighted = ops.bmm(a, encoder_outputs)

        # 将w_t的第1、2维度进行交换
        # shape = [1, batch size, enc hid dim * 2]
        weighted = weighted.transpose(1, 0, 2)

        # 将emdedded与weighted堆叠在一起，后输入进RNN层
        # rnn_input shape = [1, batch size, (enc hid dim * 2) + emb dim]
        # output shape = [seq len = 1, batch size, dec hid dim * n directions]
        # hidden shape = [n layers (1) * n directions (1) = 1, batch size, dec hid dim]
        rnn_input = ops.concat((embedded, weighted), axis=2)
        output, hidden = self.rnn(rnn_input, hidden.expand_dims(0))

        # 去除多余的第1维度
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # 将embedded，weighted和hidden堆叠起来，并输入线性层，预测下一个词
        # shape = [batch size, output dim]
        prediction = self.fc_out(ops.concat((output, weighted, embedded), axis=1))

        return prediction, hidden.squeeze(0), a.squeeze(1)


# In[12]:


class Seq2Seq(nn.Cell):
    def __init__(self, encoder, decoder, src_pad_idx, teacher_forcing_ratio):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio  # 使用teacher forcing的可能性

    def create_mask(self, src):
        """标记出每个序列中<pad>占位符的位置"""
        mask = (src != self.src_pad_idx).astype(mindspore.int32).swapaxes(1, 0)
        return mask

    def construct(self, src, src_len, trg, trg_len=None):
        """构建seq2seq模型

        Args:
            src: 源序列；shape = [src len, batch size]
            src_len: 源序列长度；shape = [batch size]
            trg: 目标序列；shape = [trg len, batch size]
            trg_len: 目标序列长度；shape = [trg len, batch size]
        """
        if trg_len is None:
            trg_len = trg.shape[0]

        #存储解码器输出
        outputs = []

        # 编码器（encoder）：
        # 输入：源序列、源序列长度
        # 输出1：编码器中所有前向与反向RNN 的隐藏状态 encoder_outputs
        # 输出2：编码器中前向与反向RNN中最后时刻的隐藏状态放入线性层后的输出 hidden
        encoder_outputs, hidden = self.encoder(src, src_len)

        #解码器的第一个输入是表示序列开始的占位符<bos>
        inputs = trg[0]

        # 标记源序列中<pad>占位符的位置
        # shape = [batch size, src len]
        mask = self.create_mask(src)

        for t in range(1, trg_len):

            # 解码器（decoder）：
            # 输入：源句子序列 inputs、前一时刻的隐藏状态 hidden、编码器所有前向与反向RNN的隐藏状态
            # 标明每个句子中的<pad>，方便计算注意力权重时忽略该部分
            # 输出：预测结果 output、新的隐藏状态 hidden、注意力权重（忽略）
            output, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)

            # 将预测结果放入之前的存储中
            outputs.append(output)

            #找出对应预测概率最大的词元
            top1 = output.argmax(1).astype(mindspore.int32)

            if self.training:
                #如果目前为模型训练状态，则按照之前设定的概率使用teacher forcing
                minval = Tensor(0, mindspore.float32)
                maxval = Tensor(1, mindspore.float32)
                teacher_force = ops.uniform((1,), minval, maxval) < self.teacher_forcing_ratio
                # 如使用teacher forcing，则将目标序列中对应的词元作为下一个输入
                # 如不使用teacher forcing，则将预测结果作为下一个输入
                inputs = trg[t] if teacher_force else top1
            else:
                inputs = top1

        # 将所有输出整合为tensor
        outputs = ops.stack(outputs, axis=0)

        return outputs


# In[13]:


input_dim = len(de_vocab.vocab())  # 输入维度
output_dim = len(en_vocab.vocab())  # 输出维度
print(input_dim, output_dim)
enc_emb_dim = 256  # Encoder Embedding层维度
dec_emb_dim = 256  # Decoder Embedding层维度
enc_hid_dim = 512  # Encoder 隐藏层维度
dec_hid_dim = 512  # Decoder 隐藏层维度
enc_dropout = 0.5  # Encoder Dropout
dec_dropout = 0.5  # Decoder Dropout
src_pad_idx = de_vocab.tokens_to_ids('<pad>')  # 德语词典中pad占位符的数字索引
trg_pad_idx = en_vocab.tokens_to_ids('<pad>')  # 英语词典中pad占位符的数字索引


attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

model = Seq2Seq(encoder, decoder, src_pad_idx, 0.5)


# In[14]:


opt = nn.Adam(model.trainable_params(), learning_rate=0.001)  # 损失函数
loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)  # 优化器


# In[15]:


def clip_by_norm(clip_norm, t, axis=None):
    """给定张量t和裁剪参数clip_norm，对t进行正则化

    使得t在axes维度上的L2-norm小于等于clip_norm。

    Args:
        t: tensor，数据类型为float
        clip_norm: scalar，数值需大于0；梯度裁剪阈值，数据类型为float
        axis: Union[None, int, tuple(int)]，数据类型为int32；计算L2-norm参考的维度，如为Norm，则参考所有维度
    """

    # 计算L2-norm
    t2 = t * t
    l2sum = t2.sum(axis=axis, keepdims=True)
    pred = l2sum > 0
    # 将加和中等于0的元素替换为1，避免后续出现NaN
    l2sum_safe = ops.select(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.select(pred, ops.sqrt(l2sum_safe), l2sum)
    # 比较L2-norm和clip_norm，如L2-norm超过阈值，进行裁剪
    # 剪裁方法：output(x) = (x * clip_norm)/max(|x|, clip_norm)
    intermediate = t * clip_norm
    cond = l2norm > clip_norm
    t_clip = ops.identity(intermediate / ops.select(cond, l2norm, clip_norm))

    return t_clip


# In[16]:


def forward_fn(src, src_len, trg):
    """前向网络"""
    print(src.shape)
    src = src.swapaxes(0, 1)
    trg = trg.swapaxes(0, 1)

    output = model(src, src_len, trg)
    output_dim = output.shape[-1]
    print(output.shape)
    output = output.view(-1, output_dim)
    print(trg.shape)
    trg = trg[1:].view(-1)
    print(output, trg)
    loss = loss_fn(output, trg)

    return loss


# 反向传播计算梯度
grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters)

def train_step(src, src_len, trg, clip):
    """单步训练"""
    loss, grads = grad_fn(src, src_len, trg)
    grads = ops.HyperMap()(ops.partial(clip_by_norm, clip), grads)  # 梯度裁剪
    opt(grads)  # 更新网络参数

    return loss


def train(dataset, clip, epoch=0):
    """模型训练"""
    model.set_train(True)
    num_batches = dataset.get_dataset_size()
    total_loss = 0  # 所有batch训练loss的累加
    total_steps = 0  # 训练步数

    with tqdm(total=num_batches) as t:
        t.set_description(f'Epoch: {epoch}')
        for src, src_len, trg in dataset.create_tuple_iterator():
            loss = train_step(src, src_len.astype(src.dtype), trg, clip)  # 当前batch的loss
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # 当前的平均loss
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


def evaluate(dataset):
    """模型验证"""
    model.set_train(False)
    num_batches = dataset.get_dataset_size()
    total_loss = 0  # 所有batch训练loss的累加
    total_steps = 0  # 训练步数

    with tqdm(total=num_batches) as t:
        for src, src_len, trg in dataset.create_tuple_iterator():
            loss = forward_fn(src, src_len, trg)  # 当前batch的loss
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # 当前的平均loss
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


# In[17]:


from mindspore import save_checkpoint

num_epochs = 10  # 训练迭代数
clip = 1.0  # 梯度裁剪阈值
best_valid_loss = float('inf')  # 当前最佳验证损失

for i in range(num_epochs):
    # 模型训练，网络权重更新
    train_loss = train(train_dataset, clip, i)
    # 网络权重更新后对模型进行验证
    valid_loss = evaluate(valid_dataset)

    # 保存当前效果最好的模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint(model, 'seq2seq.ckpt')


# In[ ]:




