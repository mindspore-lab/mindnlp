# code adapted from Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.py
import numpy as np
import torch
import torch.nn as nn
import math
from KVCache import Cache

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def get_attn_pad_mask(seq_q):
    batch_size, len_q = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_q)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(tgt_vocab_size, d_model)
        position = torch.arange(0, tgt_vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(2 * d_model) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ## pe:[tgt_len + 1, d_model]
        pe = pe.unsqueeze(0).transpose(0,1)
        ## pe:[tgt_len + 1, 1, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x, pos):
        if pos > -1:
            x = x + self.pe[pos][:x.size(0), :] # x: [batch_size, d_model]
        else:
            x = x + self.pe[:x.size(0), :] # x: [tgt_len + 1, batch_size, d_model]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k=d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, Q, batch_size, attn_mask, k_cache = None, v_cache = None):
        residual = Q
        if attn_mask is None:
            q_s = self.W_Q(Q).view(batch_size, self.n_heads, self.d_k).flatten(0,1)  # q_s: [batch_size x n_heads, d_k]
            k_s = self.W_K(Q).view(batch_size, self.n_heads, self.d_k).flatten(0,1)  # k_s: [batch_size x n_heads, d_k]
            v_s = self.W_V(Q).view(batch_size, self.n_heads, self.d_v).flatten(0,1)  # v_s: [batch_size x n_heads, d_v]
            k_cache.write(k_s)
            v_cache.write(v_s)
            context = k_cache.transmatmul(q_s)
            context = v_cache.matmul(nn.Softmax(dim=-1)(context)) # context: [batch_size x n_heads, d_v]
            context = context.contiguous().view(batch_size, -1) # context: [batch_size, n_heads x d_v]
        else:
            q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, len_q, d_k]
            k_s = self.W_K(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, len_q, d_k]
            v_s = self.W_V(Q).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, len_q, d_v]

            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, len_q, len_k]

            context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) # context: [batch_size x n_heads x len_q x d_v]
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads x d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual) # output: [batch_size, len_q, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(len(inputs.size()) - 2, len(inputs.size()) - 1)))
        output = self.conv2(output).transpose(len(inputs.size()) - 2, len(inputs.size()) - 1)
        return self.layer_norm(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, tgt_len, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.n_heads = n_heads
        self.tgt_len = tgt_len
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, dec_inputs, batch_size, dec_enc_attn_mask, block_size, k_cache, v_cache):
        if block_size > 0:
            dec_outputs = self.dec_enc_attn(dec_inputs, batch_size, None, k_cache, v_cache)
        else:
            dec_outputs = self.dec_enc_attn(dec_inputs, batch_size, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, tgt_len, d_model, d_ff, d_k, d_v, n_layers, n_heads):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(tgt_len + 1, d_model)
        self.layers = nn.ModuleList([DecoderLayer(tgt_len, d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, batch_size, k_caches, v_caches, block_size = 0, pos = -1): # dec_inputs : [batch_size] or [batch_size, tgt_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        if block_size > 0:
            dec_outputs = self.pos_emb(dec_outputs, pos)  # [batch_size, d_model]
            dec_enc_attn_mask = None
        else:
            dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1), pos).transpose(0, 1)  # [batch_size, tgt_len, d_model]
            dec_enc_attn_mask = get_attn_pad_mask(dec_inputs)

        for layer, k_cache, v_cache in list(zip(self.layers, k_caches, v_caches)):
            dec_outputs = layer(dec_outputs, batch_size, dec_enc_attn_mask, block_size, k_cache, v_cache)
        return dec_outputs

class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, tgt_len = 10, d_model = 512, d_ff = 2048, d_k = 64, d_v = 64, n_layers = 6, n_heads = 8):
        super(Transformer, self).__init__()
        self.decoder = Decoder(tgt_vocab_size, tgt_len, d_model, d_ff, d_k, d_v, n_layers, n_heads)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.tgt_len = tgt_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
    def forward(self, dec_inputs, block_size = 0):
        batch_size = dec_inputs.size()[0]
        if block_size > 0:
            k_caches = []
            v_caches = []
            dec_inputs = dec_inputs.transpose(0, 1)
            for i in range(self.n_layers):
                k_caches.append(Cache(math.ceil(batch_size * len(dec_inputs) * self.n_heads / block_size), batch_size * self.n_heads, block_size, self.d_k))
                v_caches.append(Cache(math.ceil(batch_size * len(dec_inputs) * self.n_heads / block_size), batch_size * self.n_heads, block_size, self.d_v))
            dec_output = torch.empty(self.tgt_vocab_size, self.d_model)
            dec_outputs = torch.empty(batch_size, 0, self.d_model)
            pos = 0
            dec_logits = torch.empty(batch_size, 0, self.tgt_vocab_size)
            for dec_input in dec_inputs:
                for i in range(0, dec_input.shape[0]):
                    if dec_input[i] == 0:
                        dec_input[i] = dec_logit[:, :self.tgt_vocab_size - 1].data.max(1, keepdim=True).indices[i]
                dec_output = self.decoder(dec_input, batch_size, k_caches, v_caches, block_size, pos)
                dec_logit = self.projection(dec_output) # dec_logits : [batch_size, tgt_vocab_size]
                pos = pos + 1
                dec_logits = torch.cat([dec_logits, dec_logit.unsqueeze(1)], 1)
            for i in range(self.n_layers):
                k_caches[i].delete()
                v_caches[i].delete()
        else:
            k_caches = [None] * self.n_layers
            v_caches = [None] * self.n_layers
            dec_outputs = self.decoder(dec_inputs, batch_size, k_caches, v_caches)
            dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits