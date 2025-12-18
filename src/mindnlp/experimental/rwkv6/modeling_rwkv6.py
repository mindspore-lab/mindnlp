# coding=utf-8
# Copyright 2024 BlinkDL, et al.
# Copyright 2024 yuunnn-w, et al.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2024 [Huawei Technologies Co., Ltd]
# Changes: Migrated to MindSpore interface
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore
import mindnlp
import mindtorch.nn as nn
import mindtorch.ops as ops
from typing import Tuple


class RWKV_BLOCK(nn.Module):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
        state (mindspore.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        i (int): 时间索引。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, state: mindspore.Tensor, i: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # 时间状态索引
        i0 = (2 + self.head_size) * i + 0
        i1 = (2 + self.head_size) * i + 1
        i2 = (2 + self.head_size) * i + 2
        i3 = (2 + self.head_size) * (i + 1)

        # 初始化时间状态视图
        self.state_view_channel = state[:, i0]
        self.state_view_time_1 = state[:, i1]
        self.state_view_time_2 = state[:, i2: i3, :]
        
        # 初始化层归一化
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
        self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
        self.ln2.bias = nn.Parameter(block_w['ln2.bias'])

        # 初始化激活函数
        self.silu = nn.SiLU()
        
        # 初始化注意力参数
        self.att_time_maa_x = nn.Parameter(block_w['att.time_maa_x'])
        self.att_time_maa = nn.Parameter(ops.stack([block_w['att.time_maa_w'],
                                                           block_w['att.time_maa_k'],
                                                           block_w['att.time_maa_v'],
                                                           block_w['att.time_maa_r'],
                                                           block_w['att.time_maa_g']]))
        self.att_time_maa_w1 = nn.Parameter(block_w['att.time_maa_w1'])
        self.att_time_maa_w2 = nn.Parameter(block_w['att.time_maa_w2'])
        self.att_time_decay = nn.Parameter(block_w['att.time_decay'])
        self.att_time_decay_w1 = nn.Parameter(block_w['att.time_decay_w1'])
        self.att_time_decay_w2 = nn.Parameter(block_w['att.time_decay_w2'])
        self.att_time_faaaa = nn.Parameter(block_w['att.time_faaaa'])
        self.att_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_receptance.weight = nn.Parameter(block_w['att.receptance.weight'])
        self.att_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_key.weight = nn.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_value.weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])
        self.att_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_gate.weight = nn.Parameter(block_w['att.gate.weight'])
        self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=1e-5, affine=True)
        self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
        self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])
            
        # 初始化前馈参数
        self.ffn_time_maa_k = nn.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = nn.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_receptance.weight = nn.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

    def channel_mixing(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        通道混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
        Returns:
            mindspore.Tensor: 混合后的张量，形状与输入的x相同。
        """
        sx = self.state_view_channel - x
        self.state_view_channel = x
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        r = nn.functional.sigmoid(self.ffn_receptance(xr))
        k = nn.functional.relu(self.ffn_key(xk)).pow(2)
        output = r * self.ffn_value(k)
        return output

    def time_mixing(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        时间混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, 2048]。
        Returns:
            mindspore.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.shape[0], self.n_head, self.head_size

        sx = (self.state_view_time_1 - x)
        self.state_view_time_1 = x

        xxx = x + sx * self.att_time_maa_x
        xxx = ops.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = ops.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)

        xw, xk, xv, xr, xg = ops.unbind(x.unsqueeze(1) + sx.unsqueeze(1) * (self.att_time_maa + xxx), dim=1)

        w = (self.att_time_decay + (ops.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        
        # 计算注意力机制的权重
        w = ops.exp(-ops.exp(w.view(batch_size, H, S, 1)))

        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, 1, S)
        k = self.att_key(xk).view(batch_size, H, S, 1)
        v = self.att_value(xv).view(batch_size, H, 1, S)
        g = self.silu(self.att_gate(xg))

        # 使用注意力机制更新状态
        s = self.state_view_time_2.view(batch_size, H, S, S)
        a = k @ v
        x = r @ (self.att_time_faaaa * a + s)
        s = a + w * s
        self.state_view_time_2 = s.view(batch_size, S, -1)

        # 展平x并应用组归一化和门控
        x = self.att_group_norm(x.flatten(start_dim=1)) * g

        # 应用输出层并返回结果
        return self.att_output(x)

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        模型的前向传播。
        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            mindspore.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        x = x + self.time_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x
        

class RWKV_RNN(nn.Module):
    """
    RWKV模型的RNN结构。

    Args:
        args (dict): 参数字典。
    """
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.set_train(False)

        # 加载权重
        w = mindtorch.serialization.load(args['MODEL_NAME'] + '.pth')
        
        # 将所有权重转换为float32
        self.num_layer = 0
        for k in w.keys():
            w[k] = w[k].float()
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
            if "blocks" in k: self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        
        self.num_layer += 1

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.n_embd = w['blocks.0.ln1.weight'].shape[0]
        self.head_size = self.n_embd // self.n_head
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]
        self.batch_size = args['batch_size']

        print(f"state_size: {self.state_size}") # 这里打印状态的形状
        
        # 初始化模型参数        
        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)
        self.ln0 = nn.LayerNorm(self.n_embd)
        self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
        self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])
        self.blocks = nn.ModuleList()

        # 初始化状态
        self.state = ops.zeros([self.batch_size, *self.state_size])
        
        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_BLOCK(block_w, self.n_embd, self.n_head, self.state, i))
            print(f"Loading blocks...[{i + 1}/{self.num_layer}]", end='\r')
        print()

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.n_embd, args['vocab_size'], bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

    def forward(self, token: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        模型的前向传播。
        Args:
            token (mindspore.Tensor): 输入的令牌张量。[Batch_size]
        Returns:
            mindspore.Tensor: 模型输出。
        """
        x = self.emb(token)
        x = self.ln0(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x
