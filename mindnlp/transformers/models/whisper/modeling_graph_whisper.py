# coding=utf-8
# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" MindSpore Whisper model for graph mode."""

import math
import copy
import os.path
import sys

import numpy as np
import mindspore as ms

from mindspore import nn, ops
from mindspore import Tensor, Parameter, load_param_into_net
from mindspore.ops.primitive import constexpr
from mindnlp.core.serialization import load

INF = 1. * 1e9

class WhisperGraphConfig:
    def __init__(self,
                 batch_size,
                 seq_length=80,
                 vocab_size=36560,
                 tgt_vocab_size=36560,
                 hidden_size=1024,
                 num_hidden_layers=6,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=128,
                 decoder_max_position_embeddings=128,
                 initializer_range=0.02,
                 beam_width=1,
                 max_decode_length=80,
                 length_penalty_weight=1.0,
                 sos_id=50257,
                 eos_id=50257,
                 dtype=ms.float16,
                 compute_type=ms.float16,
                 init_decode_start_ids=None
                 ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.decoder_max_position_embeddings = decoder_max_position_embeddings
        self.initializer_range = initializer_range
        self.beam_width = beam_width
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.dtype = dtype
        self.compute_type = compute_type
        self.init_decode_start_ids = init_decode_start_ids


class WhisperGraphEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 compute_type=ms.float32):
        super(WhisperGraphEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        norm = np.random.normal(0.0, embedding_size ** -0.5, [vocab_size, embedding_size]).astype(np.float32)
        self.embedding_table = Parameter(Tensor(norm, dtype=compute_type))
        self.expand = ops.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def forward(self, input_ids):
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table.value()


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class WhisperGraphEmbeddingPositionalProcessor(nn.Module):
    def __init__(self,
                 embedding_size,
                 max_position_embeddings=128,
                 dropout_prob=0.1,
                 compute_type=ms.float32):
        super(WhisperGraphEmbeddingPositionalProcessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=ms.float32)
        self.multiply = ops.Mul()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout_prob > 0
        self.expand_dims = ops.ExpandDims()
        self.position_embedding_table = Parameter(
            Tensor(position_encoding(max_position_embeddings, embedding_size), dtype=compute_type))
        self.shape = ops.Shape()

    def forward(self, word_embeddings):
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]

        # output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(word_embeddings, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class CastWrapper(nn.Module):
    def __init__(self, src_type=ms.float32, dst_type=ms.float32):
        super(CastWrapper, self).__init__()
        self.cast = ops.Cast()
        self.dst_type = dst_type

    def forward(self, x):
        return self.cast(x, self.dst_type)


class LayerPostprocess(nn.Module):
    def __init__(self,
                 dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout_prob > 0

    def forward(self, hidden_tensor, input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class WhisperGraphAttention(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_attention_heads=16,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 is_encdec_att=False,

                 do_return_2d_tensor=True,
                 compute_type=ms.float32):
        super(WhisperGraphAttention, self).__init__()
        self.batch_size = batch_size
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.has_attention_mask = has_attention_mask
        # assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = ops.Reshape()
        self.shape_from_2d = (-1, hidden_size)
        self.shape_to_2d = (-1, hidden_size)
        units = num_attention_heads * self.size_per_head
        self.query_layer = nn.Linear(hidden_size,
                                    units,
                                    bias=True, dtype=compute_type).to_float(compute_type)
        self.key_layer = nn.Linear(hidden_size,
                                  units,
                                  bias=False, dtype=compute_type).to_float(compute_type)
        self.value_layer = nn.Linear(hidden_size,
                                    units,
                                    bias=True, dtype=compute_type).to_float(compute_type)
        self.out_layer = nn.Linear(units,
                                  hidden_size,
                                  bias=True, dtype=compute_type).to_float(compute_type)

        self.matmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.multiply = ops.Mul()
        self.transpose = ops.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0, ], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = ops.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = ops.ExpandDims()
            self.sub = ops.Sub()
            self.add = ops.Add()
            self.cast = ops.Cast()

        self.is_encdec_att = is_encdec_att
        self.get_dtype = ops.DType()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = ops.Cast()
        self.shape = (-1, hidden_size)
        self.layernorm = nn.LayerNorm([hidden_size], dtype=compute_type)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

    def forward(self, input_tensor, to_tensor, attention_mask, seq_length, enc_seq_length):
        input_tensor = self.reshape(input_tensor, self.shape)
        to_tensor = self.reshape(to_tensor, self.shape)
        from_tensor = self.layernorm(input_tensor)
        if not self.is_encdec_att:
            to_tensor = from_tensor

        from_seq_length = -1
        to_seq_length = -1
        shape_from = (self.batch_size, from_seq_length, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_length, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (self.batch_size * from_seq_length, self.num_attention_heads * self.size_per_head)
            if from_seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        # attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = ops.bmm(query_layer, key_layer.swapaxes(2, 3))
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(ops.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention_scores = self.softmax_cast(attention_scores, ms.float32)
        attention_probs = self.softmax(attention_scores)
        # attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_return)
        context_layer = self.out_layer(context_layer)
        output = self.postprocess(context_layer, input_tensor)
        return output


class FeedForward(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Linear(in_channels,
                              hidden_size, dtype=compute_type).to_float(compute_type)
        self.conv2 = nn.Linear(hidden_size,
                              out_channels, dtype=compute_type).to_float(compute_type)

        self.layernorm = nn.LayerNorm([in_channels], dtype=compute_type)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = ops.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def forward(self, input_tensor):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.layernorm(input_tensor)
        output = ops.gelu(self.conv1(output))
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class WhisperGraphEncoderLayer(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(WhisperGraphEncoderLayer, self).__init__()
        self.attention = WhisperGraphAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            is_encdec_att=False,
            has_attention_mask=False,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def forward(self, hidden_states, attention_mask, seq_length):
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class WhisperGraphEncoder(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(WhisperGraphEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_hidden_layers):
            layer = WhisperGraphEncoderLayer(batch_size=batch_size,
                                             hidden_size=hidden_size,
                                             num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size,
                                             attention_probs_dropout_prob=attention_probs_dropout_prob,
                                             use_one_hot_embeddings=use_one_hot_embeddings,
                                             initializer_range=initializer_range,
                                             hidden_dropout_prob=hidden_dropout_prob,
                                             compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.layernorm = nn.LayerNorm([hidden_size], dtype=compute_type)
        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)

    def forward(self, input_tensor, attention_mask, seq_length):
        out_shape = (self.batch_size, -1, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output

        prev_output = self.layernorm(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class WhisperGraphDecoderLayer(nn.Module):

    def __init__(self,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=12,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(WhisperGraphDecoderLayer, self).__init__()
        self.self_attention = WhisperGraphAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=False,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.cross_attention = WhisperGraphAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=True,
            has_attention_mask=False,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def forward(self, hidden_states, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        # self-attention with ln, res
        attention_output = self.self_attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # cross-attention with ln, res
        attention_output = self.cross_attention(attention_output, enc_states, enc_attention_mask,
                                                seq_length, enc_seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class WhisperGraphDecoder(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(WhisperGraphDecoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for _ in range(num_hidden_layers):
            layer = WhisperGraphDecoderLayer(batch_size=batch_size,
                                             hidden_size=hidden_size,
                                             num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size,
                                             attention_probs_dropout_prob=attention_probs_dropout_prob,
                                             use_one_hot_embeddings=use_one_hot_embeddings,
                                             initializer_range=initializer_range,
                                             hidden_dropout_prob=hidden_dropout_prob,
                                             compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.layernorm = nn.LayerNorm([hidden_size], dtype=compute_type)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def forward(self, input_tensor, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask,
                                        seq_length, enc_seq_length)
            prev_output = layer_output

        prev_output = self.layernorm(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class CreateAttentionMaskFromInputMask(nn.Module):

    def __init__(self):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.batch_matmul = ops.BatchMatMul()

    def forward(self, input_mask):
        input_shape = self.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)

        input_mask = self.cast(input_mask, ms.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.batch_matmul(mask_left, mask_right)

        return attention_mask


class PredLogProbs(nn.Module):

    def __init__(self,
                 batch_size,
                 width,
                 compute_type=ms.float32,
                 dtype=ms.float32):
        super(PredLogProbs, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.compute_type = compute_type
        self.dtype = dtype

        self.reshape = ops.Reshape()
        self.matmul = ops.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = ops.Cast()

    def forward(self,
                  input_tensor,
                  output_weights,
                  seq_length):
        shape_flat_sequence_tensor = (self.batch_size * seq_length, self.width)

        input_tensor = self.reshape(input_tensor, shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class TransformerDecoderStep(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 vocab_size,
                 max_decode_length,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.3,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.3,
                 compute_type=ms.float32,
                 embedding_lookup=None,
                 embedding_processor=None):
        super(TransformerDecoderStep, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        self.tfm_embedding_lookup = embedding_lookup
        self.tfm_embedding_processor = embedding_processor
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False, dtype=compute_type).to_float(compute_type)

        self.tfm_decoder = WhisperGraphDecoder(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

        self.ones_like = ops.OnesLike()
        self.shape = ops.Shape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()

        ones = np.ones(shape=(max_decode_length, max_decode_length))
        self.future_mask = Tensor(np.tril(ones), dtype=ms.float32)

        self.cast_compute_type = CastWrapper(dst_type=compute_type)

    def forward(self, input_ids, enc_states, enc_attention_mask, seq_length):
        # process embedding
        input_embedding, embedding_tables = self.tfm_embedding_lookup(input_ids)
        input_embedding = self.tfm_embedding_processor(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        future_mask = self.future_mask[0:input_len:1, 0:input_len:1]

        input_mask = self.ones_like(input_ids)
        input_mask = self._create_attention_mask_from_input_mask(input_mask)
        input_mask = self.multiply(input_mask, self.expand(future_mask, 0))
        input_mask = self.cast_compute_type(input_mask)

        enc_attention_mask = enc_attention_mask[::, 0:input_len:1, ::]

        # call TransformerDecoder
        decoder_output = self.tfm_decoder(input_embedding, input_mask, enc_states, enc_attention_mask, -1, seq_length)

        # take the last step
        decoder_output = decoder_output[::, input_len - 1:input_len:1, ::]

        # projection and log_prob
        log_probs = self.projection(decoder_output)

        return log_probs


@constexpr
def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=ms.float32)


class WhisperGraphModel(nn.Module):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False):
        super(WhisperGraphModel, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.embedding_size = config.hidden_size
        self.tfm_embedding_lookup = WhisperGraphEmbedding(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=config.compute_type)
        self.tfm_embedding_postprocessor_for_encoder = WhisperGraphEmbeddingPositionalProcessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            compute_type=config.compute_type)
        self.tfm_embedding_postprocessor_for_decoder = WhisperGraphEmbeddingPositionalProcessor(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.decoder_max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            compute_type=config.compute_type)
        self.tfm_encoder = WhisperGraphEncoder(
            batch_size=config.batch_size,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            compute_type=config.compute_type)

        if is_training:
            self.projection = PredLogProbs(
                batch_size=config.batch_size,
                width=config.hidden_size,
                compute_type=config.compute_type,
                dtype=config.dtype)
            self.tfm_decoder = WhisperGraphDecoder(
                batch_size=config.batch_size,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_hidden_layers=config.num_hidden_layers,
                intermediate_size=config.intermediate_size,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                use_one_hot_embeddings=use_one_hot_embeddings,
                initializer_range=config.initializer_range,
                hidden_dropout_prob=config.hidden_dropout_prob,
                compute_type=config.compute_type)
        else:
            self.tfm_decoder = TransformerDecoderStep(
                batch_size=config.batch_size * config.beam_width,
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                max_decode_length=config.max_decode_length,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                use_one_hot_embeddings=False,
                initializer_range=config.initializer_range,
                hidden_dropout_prob=config.hidden_dropout_prob,
                compute_type=config.compute_type,
                embedding_lookup=self.tfm_embedding_lookup,
                embedding_processor=self.tfm_embedding_postprocessor_for_decoder)
            self.tfm_decoder = BeamSearchDecoder(
                batch_size=config.batch_size,
                seq_length=config.seq_length,
                vocab_size=config.vocab_size,
                decoder=self.tfm_decoder,
                beam_width=config.beam_width,
                length_penalty_weight=config.length_penalty_weight,
                max_decode_length=config.max_decode_length,
                sos_id=config.sos_id,
                eos_id=config.eos_id,
                init_decode_start_ids=config.init_decode_start_ids
            )

            self.tfm_decoder.add_flags(loop_can_unroll=True)
            self.tile_beam = TileBeam(beam_width=config.beam_width)
            ones = np.ones(shape=(config.batch_size, config.max_decode_length))
            self.encdec_mask = Tensor(ones, ms.float32)

        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()
        self.conv1 = nn.Conv1d(config.seq_length, config.hidden_size, kernel_size=3, padding=1, pad_mode='pad',
                               has_bias=True).to_float(config.compute_type)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1,
                               pad_mode='pad',
                               has_bias=True).to_float(config.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

    def forward(self, source_ids, source_mask, target_ids=None, target_mask=None):
        seq_length = source_ids.shape[1]

        inputs_embeds = ops.gelu(self.conv1(source_ids))
        source_ids = ops.gelu(self.conv2(inputs_embeds))

        # process source sentence
        source_ids = ops.permute(source_ids, (0, 2, 1))
        src_embedding_output = source_ids + self.tfm_embedding_postprocessor_for_encoder.position_embedding_table
        # src_embedding_output = self.tfm_embedding_postprocessor_for_encoder(source_ids)
        # attention mask [batch_size, seq_length, seq_length]
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # transformer encoder
        encoder_output = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                          None,
                                          seq_length)
        if self.is_training:
            future_mask = convert_np_to_tensor_encoder(seq_length)
            # process target sentence
            tgt_word_embeddings, embedding_tables = self.tfm_embedding_lookup(target_ids)
            tgt_embedding_output = self.tfm_embedding_postprocessor_for_decoder(tgt_word_embeddings)
            # attention mask [batch_size, seq_length, seq_length]
            tgt_attention_mask = self._create_attention_mask_from_input_mask(target_mask)
            tgt_attention_mask = self.multiply(tgt_attention_mask, self.expand(future_mask, 0))
            # transformer decoder
            decoder_output = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                              self.cast_compute_type(tgt_attention_mask),
                                              encoder_output, enc_attention_mask,
                                              -1, seq_length)
            # calculate logits and log_probs
            log_probs = self.projection(decoder_output, embedding_tables, seq_length)
            ret = log_probs
        else:
            beam_encoder_output = self.tile_beam(encoder_output)

            enc_attention_mask = self.multiply(enc_attention_mask[::, 0:1:1, ::], self.expand(self.encdec_mask, -1))

            beam_enc_attention_mask = self.tile_beam(enc_attention_mask)
            beam_enc_attention_mask = self.cast_compute_type(beam_enc_attention_mask)
            predicted_ids = self.tfm_decoder(beam_encoder_output, beam_enc_attention_mask)
            ret = predicted_ids
        return ret

    @classmethod
    def load_model_config(cls, path, lang):
        import json
        with open(os.path.join(path, "added_tokens.json"), "r") as add_tokens_file:
            add_tokens_dict = json.load(add_tokens_file)
        lang_id = add_tokens_dict["<|" + lang + "|>"]
        notimestamps_id = add_tokens_dict["<|notimestamps|>"]
        with open(os.path.join(path, "config.json"), "r") as config_file:
            config_dict = json.load(config_file)
            seq_length = config_dict['num_mel_bins']
            vocab = config_dict['vocab_size']
            hidden_size = config_dict['d_model']
            num_hidden_layers = config_dict['encoder_layers']
            num_attention_heads = config_dict['encoder_attention_heads']
            intermediate_size = config_dict['decoder_ffn_dim']
            hidden_dropout_prob = config_dict['dropout']
            attention_probs_dropout_prob = config_dict['attention_dropout']
            max_position_embeddings = config_dict['max_source_positions']
            decoder_max_position_embeddings = config_dict['max_target_positions']
            max_decode_length = config_dict['max_length']
            initializer_range = config_dict['init_std']
            sos_id = config_dict['bos_token_id']
            eos_id = config_dict['eos_token_id']
            forced_decoder_ids = config_dict['forced_decoder_ids']
            init_decode_start_ids = [sos_id, lang_id, notimestamps_id]
            model_config = WhisperGraphConfig(batch_size=1, seq_length=seq_length, vocab_size=vocab,
                                              hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                                              num_attention_heads=num_attention_heads,
                                              intermediate_size=intermediate_size,
                                              hidden_dropout_prob=hidden_dropout_prob,
                                              attention_probs_dropout_prob=attention_probs_dropout_prob,
                                              max_position_embeddings=max_position_embeddings,
                                              decoder_max_position_embeddings=decoder_max_position_embeddings,
                                              max_decode_length=max_decode_length, initializer_range=initializer_range,
                                              sos_id=sos_id, eos_id=eos_id, init_decode_start_ids=init_decode_start_ids)
            return model_config

    @classmethod
    def load_graph_model(cls, path, language, **kwargs):
        if language is None:
            print("language detected not support now!!")
            sys.exit()
        model_config = WhisperGraphModel.load_model_config(path, language)
        for key, value in kwargs.items():
            model_config.__setattr__(key, value)
        model = WhisperGraphModel(model_config, is_training=False)
        load_param_into_net(model, WhisperGraphModel.load_weight(model_config, path))
        return model

    @classmethod
    def load_weight(cls, model_config, torch_path):
        import json
        mind_param_dict = {}
        param_format = "{prefix}.{index}.{tail}"
        torch_model_dict = load(os.path.join(torch_path, "pytorch_model.bin"))
        json_config_file = os.path.join(os.path.dirname(__file__), "json", "whisper_graph_weights_mappings.json")
        with open(json_config_file, "r") as merge_json:
            merge_dict = json.load(merge_json)
            torch_encoder_layer_prefix = "model.encoder.layers"
            mind_encoder_layer_prefix = "tfm_encoder.layers"
            torch_decoder_layer_prefix = "model.decoder.layers"
            mind_decoder_layer_prefix = "tfm_decoder.tfm_decoder.layers"
            for index in range(model_config.num_hidden_layers):
                for key, value in merge_dict['encoder.layers'].items():
                    torch_key = param_format.format(index=index, prefix=torch_encoder_layer_prefix, tail=key)
                    mind_key = param_format.format(index=index, prefix=mind_encoder_layer_prefix, tail=value)
                    mind_param_dict[mind_key] = torch_model_dict[torch_key]
                # 迁移decode layers
                for key, value in merge_dict['decoder.layers'].items():
                    torch_key = param_format.format(index=index, prefix=torch_decoder_layer_prefix, tail=key)
                    mind_key = param_format.format(index=index, prefix=mind_decoder_layer_prefix, tail=value)
                    mind_param_dict[mind_key] = torch_model_dict[torch_key]

            for torch_key, mind_key in merge_dict['encoder'].items():
                mind_param_dict[mind_key] = torch_model_dict[torch_key]
            for torch_key, mind_key in merge_dict['decoder'].items():
                mind_param_dict[mind_key] = torch_model_dict[torch_key]

            if "proj_out.weight" in torch_model_dict:
                mind_param_dict["tfm_decoder.projection.weight"] = torch_model_dict["proj_out.weight"]
            else:
                mind_param_dict["tfm_decoder.projection.weight"] = torch_model_dict["model.decoder.embed_tokens.weight"]
        return mind_param_dict


class LengthPenalty(nn.Module):
    def __init__(self,
                 weight=1.0,
                 compute_type=ms.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()
        self.five = Tensor(5.0, ms.float32)
        self.six = Tensor(6.0, ms.float32)

    def forward(self, length_tensor):
        length_tensor = self.cast(length_tensor, ms.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class TileBeam(nn.Module):
    def __init__(self,
                 beam_width,
                 compute_type=ms.float32):
        super(TileBeam, self).__init__()
        self.beam_width = beam_width
        self.expand = ops.ExpandDims()
        self.tile = ops.Tile()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def forward(self, input_tensor):
        shape = self.shape(input_tensor)
        input_tensor = self.expand(input_tensor, 1)
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape) - 1):
            tile_shape = tile_shape + (1,)
        output = self.tile(input_tensor, tile_shape)
        out_shape = (shape[0] * self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)
        return output


class BeamSearchDecoder(nn.Module):

    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 decoder,
                 beam_width=4,
                 length_penalty_weight=1.0,
                 max_decode_length=128,
                 sos_id=50257,
                 eos_id=50256,
                 init_decode_start_ids=None):
        super(BeamSearchDecoder, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.decoder = decoder
        self.add = ops.Add()
        self.expand = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.shape_flat = (-1,)
        self.shape = ops.Shape()

        self.zero_tensor = Tensor(np.zeros([batch_size, beam_width]), ms.float32)
        self.ninf_tensor = Tensor(np.full([batch_size, beam_width], -INF), ms.float32)

        self.select = ops.Select()
        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.topk = ops.TopK(sorted=True)
        self.floor_div = ops.FloorDiv()
        self.vocab_size_tensor = Tensor(self.vocab_size, ms.int32)
        self.real_div = ops.RealDiv()
        self.equal = ops.Equal()
        self.eos_ids = Tensor(np.full([batch_size, beam_width], eos_id), ms.int32)

        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = Tensor(beam_ids, ms.int32)
        batch_ids = np.arange(batch_size * beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = Tensor(batch_ids, ms.int32)
        self.concat = ops.Concat(axis=-1)
        self.gather_nd = ops.GatherNd()

        self.greater_equal = ops.GreaterEqual()
        self.sub = ops.Sub()
        self.cast = ops.Cast()
        self.zeroslike = ops.ZerosLike()

        # init inputs and states
        self.start_ids = Tensor(np.full([batch_size * beam_width, 1], sos_id), ms.int32)
        self.init_seq = Tensor(np.full([batch_size, beam_width, 1], sos_id), ms.int32)
        init_scores = np.tile(np.array([[0.] + [-INF] * (beam_width - 1)]), [batch_size, 1])
        self.init_scores = Tensor(init_scores, ms.float32)
        self.init_finished = Tensor(np.zeros([batch_size, beam_width], dtype=np.bool_))
        self.init_length = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
        self.length_penalty = LengthPenalty(weight=length_penalty_weight)
        self.one = Tensor(1, ms.int32)

        self.forced_decoder_ids_len = len(init_decode_start_ids)
        if self.forced_decoder_ids_len > 0:
            forced_decoder_ids_init = Tensor(init_decode_start_ids, dtype=ms.int32).reshape(1, -1)
            self.start_ids = ops.repeat_elements(forced_decoder_ids_init, batch_size * beam_width, axis=0)
            self.init_seq = self.start_ids.reshape(batch_size, beam_width, -1)
            self.init_length = Tensor(np.full([batch_size, beam_width], self.forced_decoder_ids_len), ms.int32)

    def one_step(self, cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                 state_seq, state_finished, state_length):
        log_probs = self.decoder(cur_input_ids, enc_states, enc_attention_mask, self.seq_length)
        # 时间戳生成
        log_probs = self.reshape(log_probs, (self.batch_size * self.beam_width, self.vocab_size))

        log_probs = ops.log_softmax(log_probs, axis=-1)
        log_probs = self.reshape(log_probs, (self.batch_size, self.beam_width, self.vocab_size))

        # select topk indices
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1))

        # mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))

        # reshape scores to [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)
        # select topk
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = self.cast(self.greater_equal(temp, 0), ms.int32)
            beam_indices = beam_indices + res
        word_indices = topk_indices - beam_indices * self.vocab_size_tensor
        # ======================================================================

        # mask finished indices
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        ###### put finished sequences to the end
        # sort according to scores with -inf for finished beams
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)
        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # update
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        beam_indices = self.gather_nd(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd(topk_scores, tmp_gather_indices)

        ###### generate new beam_search states
        # gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd(state_length, gather_indices)

        # concat seq
        seq = self.gather_nd(state_seq, gather_indices)
        state_seq = self.concat((seq, self.expand(word_indices, -1)))

        # new finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores

        cur_input_ids = self.reshape(state_seq, (self.batch_size * self.beam_width, -1))
        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length

    def forward(self, enc_states, enc_attention_mask):
        cur_input_ids = self.start_ids
        state_log_probs = self.init_scores
        state_seq = self.init_seq
        state_finished = self.init_finished
        state_length = self.init_length

        for index in range(self.max_decode_length - self.forced_decoder_ids_len):
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length = self.one_step(
                cur_input_ids, enc_states, enc_attention_mask, state_log_probs, state_seq, state_finished, state_length)

        penalty_len = self.length_penalty(state_length)
        log_probs = self.real_div(state_log_probs, penalty_len)

        _, top_beam_indices = self.topk(log_probs, self.beam_width)
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(top_beam_indices, -1)))
        predicted_ids = self.gather_nd(state_seq, gather_indices)
        predicted_ids = predicted_ids[::, 0:1:1, ::]
        return predicted_ids
