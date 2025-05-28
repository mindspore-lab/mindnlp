import pytest
from .timm_layers import (
    Mlp,
    AttentionPoolLatent,
    DropPath,
    LayerType,
    PatchDropout,
    PatchEmbed,
    resample_abs_pos_embed
)
import mindspore as ms
import numpy as np

# 测试 Mlp 类
def test_mlp():
    mlp = Mlp(in_features=10, hidden_features=20, out_features=30)
    x = ms.Tensor(np.random.randn(1, 10), dtype=ms.float32)
    output = mlp(x)
    assert output.shape == (1, 30)

# 测试 AttentionPoolLatent 类
def test_attention_pool_latent():
    attn_pool = AttentionPoolLatent(in_features=10, out_features=20)
    x = ms.Tensor(np.random.randn(1, 100, 10))
    output = attn_pool(x)
    assert output.shape == (1, 20)

# 测试 DropPath 类
def test_drop_path():
    drop_path = DropPath(drop_prob=0.5)
    x = ms.Tensor(np.random.randn(1, 10))
    output = drop_path(x)
    # 由于随机性，这里只检查输出的形状
    assert output.shape == x.shape

# 测试 PatchDropout 类
def test_patch_dropout():
    patch_dropout = PatchDropout(prob=0.5)
    x = ms.Tensor(np.random.randn(1, 100, 10))
    output = patch_dropout(x)
    # 由于随机性，这里只检查输出的形状
    assert output.shape == x.shape

# 测试 PatchEmbed 类
def test_patch_embed():
    patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    x = ms.Tensor(np.random.randn(1, 3, 224, 224))
    output = patch_embed(x)
    assert output.shape == (1, 196, 768)

# 测试 resample_abs_pos_embed 函数
def test_resample_abs_pos_embed():
    posemb = ms.Tensor(np.random.randn(1, 100, 768))
    new_size = [50, 50]
    output = resample_abs_pos_embed(posemb, new_size)
    assert output.shape == (1, 2500, 768)

if __name__ == '__main__':
    pytest.main()
