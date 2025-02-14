# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from dataclasses import dataclass, field
from typing import List

import mindspore
# import mindspore.ops as ops
import mindnlp.core.ops as ops
import mindnlp.core.nn as nn
import mindspore.common.dtype as mstype
import mindnlp.core.nn.functional as F
from mindnlp.core.nn import Parameter
from mindnlp.core import Tensor
from mindspore.common.initializer import initializer, Uniform
from functools import partial
import numpy as np


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

def normalize_np(x, p=2, dim=1, eps=1e-12):
    x = x.asnumpy()
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)
    y = x / norm
    y = mindspore.Tensor(y)
    return y

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        z_channels=256,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        left= mindspore.Tensor(-1.0 / self.n_e, dtype=self.embedding.weight.dtype)
        right= mindspore.Tensor(1.0 / self.n_e, dtype=self.embedding.weight.dtype)
        # self.embedding.weight = Parameter(initializer(Uniform(1.0 / self.n_e), self.embedding.weight.shape, self.embedding.weight.dtype))
        nn.init.uniform_(self.embedding.weight, -1.0 / self.n_e, 1.0 / self.n_e)
        # uniform(self.embedding.weight.shape, left, right,self.embedding.weight.dtype, seed=0)
        # mindspore.common.initializer.Uniform((-1.0 / self.n_e, 1.0 / self.n_e)).apply(self.embedding.weight)
        if self.l2_norm:
            self.embedding.weight = Parameter(F.normalize(
                self.embedding.weight, p=2, dim=-1
            ))
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(ops.zeros(65536)))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = ops.permute(z, (0, 2, 3, 1))
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = (
            ops.sum(z_flattened**2, dim=1, keepdim=True)
            + ops.sum(embedding**2, dim=1)
            - 2
            * ops.einsum(
                "bd,dn->bn", z_flattened, ops.einsum("n d -> d n", embedding)
            )
        )

        min_encoding_indices = ops.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        # compute loss for embedding
        if self.training:
            vq_loss = ops.mean((z_q - ops.stop_gradient(z)) ** 2)
            commit_loss = self.beta * ops.mean((ops.stop_gradient(z_q) - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + ops.stop_gradient(z_q - z)

        # reshape back to match original input shape
        z_q = ops.einsum("b h w c -> b c h w", z_q)

        return (
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )
   
    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            print("self.embedding.weight.dtype:", self.embedding.weight.dtype)
            embedding = F.normalize(self.embedding.weight.astype(mindspore.float16), p=2, dim=-1)
            # embedding = normalize_np(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        print("AttnBlock forward")
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c  (1, 576, 512)
        k = k.reshape(b, c, h * w)  # b,c,hw
        # q = q.permute(0, 2, 1) # (1, 576, 512)
        print("q.shape:",q.shape) # q.shape: (1, 576, 512)
        print("k.shape:",k.shape) # k.shape: (1, 512, 576) (576,512)
        w_ = ops.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        print("w_.shape:",w_.shape)
        print("v.shape:",v.shape)
        h_ = ops.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        print("h_.dtype:",h_.dtype)

        h_ = self.proj_out(h_.astype(mindspore.float16))

        return x + h_


def nonlinearity(x):
    # swish
    return x * ops.sigmoid(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        assert False, "not implement batch Norm"
        # return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        if x.dtype != mindspore.float32:
            x = F.interpolate(x.astype(mstype.float32), scale_factor=2.0, mode="nearest", recompute_scale_factor=True).astype(mstype.float16)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = ops.mean(target_probs, axis=0)
    avg_entropy = -ops.sum(avg_probs * ops.log(avg_probs + 1e-5))
    sample_entropy = -ops.mean(ops.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(
            config.codebook_embed_dim, config.z_channels, 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_16(**kwargs):
    return VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs
        )
    )


VQ_models = {"VQ-16": VQ_16}
if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    np_tokens = np.load('/home/xuhang/np_token.npy')
    print('np_tokens shape:', np_tokens.shape)
    vq_model = VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]
        )
    )
    # vq_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[
    #     parallel_size, 8, img_size//patch_size, img_size//patch_size])
    # ret = vq_model.decode_code(ms.Tensor(np_tokens).astype(ms.int32), shape=[
    #     16, 8, 384//16,  384//16])
    # print(ret.shape) # 16, 3, 384, 384
    ms_token = np.load('/home/xuhang/ms_token.npy')
    print('ms_token shape:', ms_token.shape)
    # vq_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[
    #     parallel_size, 8, img_size//patch_size, img_size//patch_size])
    ret = vq_model.decode_code(Tensor(ms_token).astype(ms.int32), shape=[
        2, 8, 384//16,  384//16])
    print(ret.shape) #  2, 3, 384, 384
    ret = ret.astype(ms.float32).asnumpy().transpose(0, 2, 3, 1)