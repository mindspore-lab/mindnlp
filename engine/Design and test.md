#### 一、技术调研

KV Cache 是一种用于加速 Transformer 等大型语言模型（LLM）自回归推理的关键优化技术，通常用于加速 Transformer 的 Decoder 部分，本次任务在 Decoder-only 结构的 Transformer 的基础之上，合理设计单机八卡 KV Cache 管理代码，对 Transformer 进行加速。

##### 1. Transformer 的多头注意力机制

在 Transformer 被提出的论文 《Attention is All You Need》 中详细介绍了多头注意力机制(Multi-Head Attention)层的基本结构。

一般而言，Transformer 可以对多批(batch)数据进行处理。在这里，我们认为多头注意力机制层的输入是 $b$ 个 $l$ 行 $d_m$ 列矩阵组成的矩阵组 $(\textbf X_0,\textbf X_1,\dots,\textbf X_{b-1})$，每个矩阵代表一批数据，矩阵的每一行代表数据的一个词元(token)。

首先，代码会对输入的数据进行“分头”操作。具体而言，是训练三组矩阵 $(\textbf W_{Q,0},\textbf W_{Q,1},\dots,\textbf W_{Q,h-1}),(\textbf W_{K,0},\textbf W_{K,1},\dots,\textbf W_{K,h-1}),(\textbf W_{V,0},\textbf W_{V,1},\dots,\textbf W_{V,h-1})$，其中第一组和第二组是 $d_m$ 行 $d_k$ 列矩阵，第三组是 $d_m$ 行 $d_v$ 列矩阵，并对每一个在 $[0,b)$ 内的自然数 $i$ 和 $[0,h)$ 的自然数 $j$，计算

$$\textbf Q_{ib+j}=\textbf X_i\textbf W_{Q,j}\\\textbf K_{ib+j}=\textbf X_i\textbf W_{K,j}\\\textbf V_{ib+j}=\textbf X_i\textbf W_{V,j}$$

接着，对每一个在 $[0,b)$ 内的自然数 $i$ 和 $[0,h)$ 的自然数 $j$，计算 $\textbf C_{i,j}=Att(\textbf Q_{ib+j},\textbf K_{ib+j},\textbf V_{ib+j},\textbf M)$，其中 $\textbf M$ 被称为掩码矩阵，对于掩码多头注意力层而言， $\textbf M$ 是主对角线上方（不包括主对角线）全为 $-\infty$，其余部分全为 $0$ 的矩阵，而函数 $Att(\textbf Q,\textbf K,\textbf V,\textbf M)$ 的表达式为

$$Att(\textbf Q,\textbf K,\textbf V,\textbf M)=softmax\left(\frac{\textbf Q\textbf K^T}{\sqrt{d_k}}+\textbf M\right)\textbf V$$

其中 $softmax(\textbf A)$ 的第 $i$ 行是行向量 $softmax(\textbf a_i)$，其中 $\textbf a_i$ 是 $\textbf A$ 的第 $i$ 行向量，而

$$softmax(a_0,a_1,\dots,a_m)=\frac{(e^{a_0},e^{a_1},\dots,e^{a_{m-1}})}{\overset{m-1}{\underset{i=0}\sum}e^{a_i}}$$

然后，生成 $b$ 个 $l$ 行 $h\times d_v$ 列矩阵 $\textbf O_{0},\textbf O_{1},\dots,\textbf O_{b-1}$，其中，矩阵 $\textbf O_{i}$ 的第 $j$ 行是分块矩阵 $[\textbf C_{i,0}\vdots \textbf C_{i,1}\vdots \cdots\vdots \textbf C_{i,h-1}]$ 的第 $j$ 行

最后，训练 $h\times d_v$ 行 $d$ 列矩阵 $\textbf W$ 并输出 $b$ 个 $l$ 行 $d$ 列矩阵组成的矩阵组 $(\textbf Y_0,\textbf Y_1,\dots,\textbf Y_{b-1})$，其中，对每一个在 $[0,b)$ 内的自然数 $i$，都有 $\textbf Y_i=\textbf O_i\textbf W$

##### 2. Decoder-only 结构下 Transformer 的优化可行性

《Attention is All You Need》 的 Transformer 是 Encoder-Decoder 的，而 Encoder 的掩码矩阵无法屏蔽未来的信息，所以这里直接使用 Decoder-only 结构下的 Transformer

我们认为，一个句子是由若干词元(token)组成的，每个 token 可以赋予一个编码，句子就变成了一个以编码为字符的字符串。Decoder-only 结构下的 Transformer 以这样的 $b$ 批字符串组 $(s_0,s_1,\dots,s_{b-1})$ 为输入，其中每个字符串的长度都是 $l$，输出是 $b$ 个 $l$ 行 $v$ 列矩阵组成的矩阵组，第 $i(0\leq i<b)$ 个矩阵的第 $j(0\leq j<l)$ 行第 $k(0\leq k<v)$ 列代表 $s_i$ 的长度为 $j$ 的前缀的下一个字符编码为 $k$ 的概率，其中 $v$ 代表 token 总数

Decoder-only 结构下的 Transformer 可以概括成下图所示

<img src=50e8005addb244c88d30f9daccaabf8a.png>

其中 Input Embedding 层是用于将 token 编码转化为向量，该层训练定义域为 $\{0,1,\dots,v\}$，值域为 $d_m$ 维向量的函数 $Emb(x)$，其时间复杂度是线性的。Input Embedding 层输出 $b$ 个 $l$ 行 $d_m$ 列矩阵组成的矩阵组，矩阵组的第 $i(0\leq i<b)$ 个矩阵的第 $j(0\leq j<l)$ 行是 $Emb(c_{i,j})$，其 $c_{i,j}$ 是 $s_i$ 的第 $j$ 个字符。Linear 层、LayerNorm 层、Feed Forward 层和 Softmax 层以 $b$ 个 $l$ 行的矩阵组成的矩阵组作为输入，输出是 $b$ 个 $l$ 行矩阵组成的矩阵组，其第 $i(0\leq i<b)$ 个矩阵的第 $j(0\leq j<l)$ 行只由输入的第 $i(0\leq i<b)$ 个矩阵的第 $j(0\leq j<l)$ 行经线性时间复杂度的运算得到。

通常对 Transformer 推理时，先将起始 token 对应编码作为输入字符串的开头输入到 Transformer 中，再用输出的相应矩阵的第 $1$ 行中的最大的那一列的列号更新输入的相应字符串的第 $2$ 个字符，然后再次输入到 Transformer 中，接着用输出的相应矩阵的第 $2$ 行中的最大的那一列的列号更新输入的相应字符串的第 $3$ 个字符，再次输入到 Transformer 中，依次类推，直到字符串的最后一个字符被用于更新。

不难得出，从第 $2$ 次调用 Transformer 时，对每一个在 $[0,b)$ 内的自然数 $i$ 和 $[0,h)$ 的自然数 $j$，每一个 Transformer Block 的多头注意力机制层的 $\textbf Q_{ib+j},\textbf K_{ib+j},\textbf V_{ib+j}$ 只比上一次调用 Transformer 多出最后一行，并且这一行只与 $\textbf X_i$ 的最后一行有关。不难想到上述推理过程中，第 $i(0\leq i<l)$ 次调用 Transformer 只要输入每一批次的第 $i$ 个词元编码，并把多头注意力机制层的 $\textbf K_{ib+j},\textbf V_{ib+j}$ 存起来，就能快速得到结果。这便是 KV Cache 技术的核心原理。

Transformer 的训练需要输入完整的字符串，所以无法通过 KV Cache 技术优化。

##### 3. vLLM 和 LightLLM 的内存管理方案

vLLM 和 LightLLM 都致力于解决大语言模型推理中 KV Cache 的内存管理问题，但两者采取了截然不同的技术路径。vLLM 的 PagedAttention 和 LightLLM 的 TokenAttention 是解决 KV Cache 的内存管理问题通用的两种方案。

PagedAttention 核心思想是将连续的 KV Cache 逻辑空间，映射到非连续的物理内存块中，就像操作系统的虚拟内存管理一样。PagedAttention 将若干个 token 对应的 $\textbf K_{ib+j}$ 矩阵行向量、$\textbf V_{ib+j}$ 矩阵行向量存储在一个块中，并维护一个映射表(block table)存储每个行向量在块中的位置。这种方案大幅减少外部碎片，但块内可能仍有少量内部碎片。

TokenAttention 预先分配一大块连续的显存存储 $\textbf K_{ib+j},\textbf V_{ib+j}$，直接通过 token 的编号找到对应的 $\textbf K_{ib+j}$ 矩阵行向量、$\textbf V_{ib+j}$ 矩阵行向量的位置。这种方案零内存浪费，实现理论上的最优内存利用，但是可能会产生无法找到足够大的空间的情况。

#### 二、方案设计

##### 1. Decoder-only 结构下 Transformer 设计

代码改编自 graykode 的代码。代码除了将原来 Transformer 改成 Decoder-only 的结构之外，还将 `Transformer` 类分成训练模式和推理模式。代码中的 `Transformer` 类用于打包各个训练层并用于训练，其 `forward` 函数输入 $b$ 行 $l$ 列的二维张量 `dec_inputs` 和变量 `block_size`，`dec_inputs` 的第 $i(0\leq i<b)$ 行第 $j(0\leq j<l)$ 列代表 $s_i$ 的第 $j$ 个字符。当 `block_size` 为 $0$ 时，开启训练模式，训练过程大致和前文所述的一致。当 `block_size` 大于 $0$ 时，开启推理模式。

推理模式为 Transformer Block 的每一个多头注意力机制层使用 `Cache` 类开辟两个内存空间 `k_cache` 和 `v_cache` 中，分别存储 $\textbf K_{ib+j}$ 和 $\textbf V_{ib+j}$，其中的参数 `num_blocks`、`num_heads`、`block_size` 和 `head_dim` 分别代表申请的内存块总数 $n=\lceil\frac{blh}t\rceil$、批次数与每层的分头数之积 $bh$、每块存储的行向量数 $t$、每个行向量对应的维数 $d$。这些内存空间会在使用完毕之后通过 `delete` 函数释放。行向量的每一位都是 $32$ 位浮点数，每个浮点数占用的空间为 $f=4B$。假设 Transformer 共有 $L$ 层 Transformer Block，那么 Transformer 推理模式因 KV Cache 占用的总空间为

$$m=2Lntdf$$

推理模式下 `Transformer` 类的 `forward` 函数调用 Transformer $l$ 次，第 $i(0\leq i<l)$ 次输入 `dec_inputs` 的第 $i$ 列向量 $(c_0,c_1,\dots,c_b)$，如果这一列的第 $j(0\leq j<b)$ 列是 $0$，那么这个位置会被替换为上一次输出的第 $i$ 行中的最大的那一列的列号。Input Embedding 层改为输出一个 $b$ 行 $d_m$ 列矩阵，矩阵的第 $i(0\leq i<b)$ 行是 $Emb(c_{i})$。Linear 层、LayerNorm 层、Feed Forward 层和 Softmax 层基本不变，只是输入和输出都变为 $b$ 行矩阵了。此时 Transformer 最终会输出一个 $b$ 行 $v$ 列矩阵。`forward` 函数维护 $b$ 个 $v$ 列矩阵组成的矩阵组，每次调用 Transformer 时会将输出的第 $i(0\leq i<b)$ 行接在矩阵组的第 $i$ 个矩阵后面，并输出这个矩阵组。

推理模式下多头注意力机制层先输入 $b$ 行 $d_m$ 列矩阵 $X$，并训练 $b$ 行 $h\times d_m$ 列矩阵 $W_Q,W_K,W_V$，并对每一个在 $[0,h)$ 内的自然数 $i$，计算

$$\textbf Q=\textbf X\textbf W_{Q}\\\textbf K=\textbf X\textbf W_{K}\\\textbf V=\textbf X\textbf W_{V}$$

再计算 $flat(\textbf Q),flat(\textbf K),flat(\textbf V)$，其中 $flat(\textbf A)$ 是将 $b$ 行 $h\times d_m$ 列 $\textbf A$ 变为 $b\times h$ 行 $d_m$ 列向量，对所有在 $[0,b)$ 内的自然数 $i$ 和 $[0,h)$ 的自然数 $j$，$\textbf A$ 的第 $i$ 行第 $jd_m,(jd_m+1),\dots,[jd_m+(d_m-1)]$ 个元素分别是 $flat(\textbf A)$ 第 $ih+j$ 行第 $0,1,\dots,(d_m-1)$ 个元素。

然后将 $flat(\textbf K)$ 和 $flat(\textbf V)$ 通过 `Cache` 的 `write` 函数分别存入 `k_cache` 和 `v_cache` 中，其效果是将 $\textbf K$ 和 $\textbf V$ 的第 $(ib+j)(0\leq i<b,0\leq j<h)$ 行并入到 $\textbf K_{ib+j}$ 和 $\textbf V_{ib+j}$ 的末尾。

接着，对每一个在 $[0,b)$ 内的自然数 $i$ 和 $[0,h)$ 的自然数 $j$，计算 $C=Att(flat(\textbf Q),\textbf K_{ib+j},\textbf V_{ib+j},\textbf M)$，计算时用到 `Cache` 的 `transmatmul` 函数和 `matmul` 函数，这两个函数用于从内存中取出需要的行向量组成若干个矩阵参与矩阵乘法运算。

最后，训练 $h\times d_v$ 行 $d$ 列矩阵 $\textbf W$ 并输出 $\textbf Y=flat^{-1}(\textbf C)\textbf W$，其中 $flat^{-1}(\textbf A)$ 是 $flat(\textbf A)$ 的反函数。

##### 2. KV Cache 的管理逻辑

`Cache` 的 `write`、`transmatmul` 和 `matmul` 函数需要写入或读出相应的矩阵。参考 vLLM 的 PagedAttention，通过按块存储的方式，设计了 `BlockAllocator` 类申请、写入、读出向量和 `MetadataEngine` 类记录请求与物理块的映射关系。

`BlockAllocator` 类初始化时统一使用 `mindspore.Parameter` 算子申请 $n$ 个物理块，每个块是一个 $t$ 行 $d$ 列矩阵，使每个物理块能够存储在不同的位置上。定义 `alloc` 函数使用 `mindspore.ops.ScatterNdUpdate` 算子逐行写入矩阵，其输入 $bh$ 行 $2$ 列的矩阵 `indices` 和待存储的矩阵 `A`，对每一个在 $[0,b)$ 内的自然数 $i$，`indices[i][0]` 和 `indices[i][1]` 分别是 `A` 的第 $i$ 行存储的物理块号和行号。定义 `get` 函数使用 `mindspore.ops.gather` 算子逐行读出矩阵，函数输入 $bh$ 行 $2$ 列的矩阵 `indices`，对每一个在 $[0,b)$ 内的自然数 $i$，`indices[i][0]` 和 `indices[i][1]` 分别输出矩阵的第 $i$ 行存储的物理块号和行号。

`MetadataEngine` 类维护一张映射表 `block_table`，对每一个在 $[0,l)$ 内的自然数 $x$ 和在 $[0,bh)$ 内的自然数 $y$ ，`block_table[x][y]` 表示矩阵 $K_y$ 或 $V_y$ 的第 $x$ 行。两个变量 `i` 和 `j`，代表第 `i` 个物理块的第 `j` 行是空的。定义 `alloc` 函数依次为 $K_0,K_1,\dots,K_{bh}$ 或 $V_0,V_1,\dots,V_{bh}$ 分配一行空间，每分配一行空间，就更新 `block_table` 的内容，并让 `i` 和 `j` 指向下一个空行。定义 `get` 函数输入一个变量 `x`，输出 `block_table[0][x]`、`block_table[1][x]`、`block_table[2][x]` 等 `list` 合并后的矩阵，代表 $K_x$ 或 $V_x$ 存储在内存中的位置。

#### 三、代码测试

之后补充。