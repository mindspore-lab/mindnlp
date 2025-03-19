# bit微调

实现了"HorcruxNo13/bit-50"模型在"dpdl-benchmark/oxford_flowers102"数据集上的微调实验。
任务链接在https://gitee.com/mindspore/community/issues/IAUPCI
transformers+pytorch+3090的benchmark是自己编写的，仓库位于https://github.com/outbreak-sen/Bit_flowers102_Finetune
更改代码位于llm/finetune/bit，只包含mindnlp+mindspore的
实验结果如下

## 硬件

资源规格：NPU: 1*Ascend-D910B(显存: 64GB), CPU: 24, 内存: 192GB

智算中心：武汉智算中心

镜像：mindspore_2_5_py311_cann8

torch训练硬件资源规格：Nvidia 3090

## 模型与数据集

模型："HorcruxNo13/bit-50"

数据集："dpdl-benchmark/oxford_flowers102"

## Eval Loss Values 表格

| Epoch | mindNLP       | torch         |
|-------|---------------|---------------|
| 1     | 3.5184175968  | 4.6460494995  |
| 2     | 1.7758612633  | 4.2146801949  |
| 3     | 0.9314232469  | 3.8055384159  |
| 4     | 0.6095938683  | 3.4315345287  |
| 5     | 0.4878421128  | 3.1143600941  |
| 6     | 0.4401741028  | 2.8422958851  |
| 7     | 0.4239776731  | 2.6192340851  |
| 8     | 0.4162144363  | 2.4506986141  |
| 9     | 0.4113974869  | 2.3450050354  |
| 10    | 0.4095760584  | 2.2997686863  |

## Test Accuracy 表格

| Epoch | mindNLP       | torch         |
|-------|---------------|---------------|
| 1     | 0.9219        | 0.6225        |

## 图片分类测试

问题来自评估数据集的第一个问题，微调后看起来效果不太好。

* 问题输入：
  dataset['test'][0]['image']
* 真实标签：
  26  
* mindnlp未微调前的回答：
  25
* mindnlp微调后的回答：
  26
* torch微调前的回答：
  41
* torch微调后的回答：
  26