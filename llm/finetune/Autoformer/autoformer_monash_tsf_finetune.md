# Autoformer Mindnlp 微调

- Autoformer模型微调任务链接：[【开源实习】autoformer模型微调 · Issue #IAUOTL · MindSpore/community - Gitee.com](https://gitee.com/mindspore/community/issues/IAUOTL)
- 实现了huggingface/autoformer-tourism-monthly 基准权重 在 [monash_tsf/tourism_monthly] 数据集上的微调
- base model: [huggingface/autoformer-tourism-monthly · Hugging Face](https://huggingface.co/huggingface/autoformer-tourism-monthly)
- dataset: [Monash-University/monash_tsf · Datasets at Hugging Face](https://huggingface.co/datasets/Monash-University/monash_tsf)

------

# Requirments
## Pytorch 

- GPU: RTX 4070ti 12G
- cuda: 11.8
- Python version: 3.10
- torch version: 2.5.0
- transformers version : 4.47.0
- accelerate: 0.27.0
- gluonts: 0.14.0
- datasets: 2.16.0
- evaluate: 0.4.0
- numpy: 1.26.4
- pandas: 2.1.0
- scipy: 1.11.0

## Mindspore 启智社区 Ascend910B算力资源
- Ascend: 910B
- python: 3.9
- mindspore: 2.5.0
- mindnlp: 0.4.1
- gluonts: 0.16.0
- datasets: 3.5.0
- evaluate: 0.4.3
- numpy: 1.26.4
- pandas: 2.2.3
- scipy: 1.13.1

 

---

## 修改内容

### Ascend

源码中**modeling_autoformer.py**文件中 **padding_mode** = **circular** 改成 **padding_mode** = **replicate**

### CPU/GPU

源码中**modeling_autoformer.py** 922行 roll操作 在gpu和cpu上没有实现

修改源码中922行语句：

```python
value_states_roll_delay = value_states.roll(shifts=-int(top_k_delays_index[i]), dims=1)
```

改成

```python
value_states_roll_delay = custom_roll(
    value_states,
    shifts=-int(top_k_delays_index[i].asnumpy())，# 转换为Python整数
    dim=1
)
```

并且在前面添加一个用于替代的roll函数的自定义方法

```python
def custom_roll(tensor, shifts, dim):
    """
    Custom implementation of cyclic shift along specified dimension
    
    Args:
        tensor: Input tensor to be shifted
        shifts: Number of positions to shift 
            (positive = right shift, negative = left shift)
        dim: Dimension index along which to perform shift
    
    Returns:
        Tensor with elements cyclically shifted along specified dimension
    """
    # Handle cases where shifts exceed dimension length
    dim_size = tensor.shape[dim]
    shifts = shifts % dim_size  # Ensure shifts are within valid range
    
    if shifts == 0:
        return tensor
    
    # Split tensor into two parts and swap their order
    if shifts > 0:
        # Right shift: keep last 'shifts' elements and move to front
        part1 = ops.narrow(tensor, dim, 0, dim_size - shifts)
        part2 = ops.narrow(tensor, dim, dim_size - shifts, shifts)
    else:
        # Left shift: keep first '|shifts|' elements and move to end
        shifts = abs(shifts)
        part1 = ops.narrow(tensor, dim, shifts, dim_size - shifts)
        part2 = ops.narrow(tensor, dim, 0, shifts)
    
    # Concatenate the reversed parts
    return ops.cat((part2, part1), dim)
```



----

## 微调结果

### Mindspore

| Epoch | Loss              |
| ----- | ----------------- |
| 0     | 7.546689510345459 |
| 1     | 7.772482395172119 |
| 2     | 7.14789342880249  |
| 3     | 7.49253511428833  |
| 4     | 7.337801456451416 |
| 5     | 6.960692882537842 |
| 6     | 8.312647819519043 |
| 7     | 6.90599250793457  |
| 8     | 7.212374210357666 |
| 9     | 7.506921291351318 |

------

### Pytorch

| Epoch | Loss               |
| ----- | ------------------ |
| 0     | 7.412668228149414  |
| 1     | 7.8263068199157715 |
| 2     | 7.839258670806885  |
| 3     | 8.043777465820312  |
| 4     | 8.08508586883545   |
| 5     | 7.503101825714111  |
| 6     | 7.824302673339844  |
| 7     | 7.399034023284912  |
| 8     | 7.122222900390625  |
| 9     | 7.612663269042969  |
