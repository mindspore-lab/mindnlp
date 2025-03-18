# bigbird_pegasus微调
实现了bigbird_pegasus模型在google/Synthetic-Persona-Chat数据集上的微调实验。
任务链接在https://gitee.com/mindspore/community/issues/IAUPBF
transformers+pytorch+3090的benchmark是自己编写的，仓库位于https://github.com/outbreak-sen/bigbird_pegasus_finetune
更改代码位于llm/finetune/bigbird_prgasus，只包含mindnlp+mindspore的
实验结果如下
## Loss Values 表格

| 序号 | MindNLP    | PyTorch |
|------|-----------|---------|
| 1    | 0.1826    | 7.6556  |
| 2    | 0.1614    | 0.5960  |
| 3    | 0.1435    | 0.4145  |
| 4    | 0.1398    | 0.3022  |
| 5    | 0.1344    | 0.2555  |
| 6    | 0.1263    | 0.2357  |
| 7    | 0.1200    | 0.2247  |
| 8    | 0.1147    | 0.2166  |
| 9    | 0.1105    | 0.2107  |
| 10   | 0.1082    | 0.2075  |

## Eval Loss Values 表格

| 序号 | MindNLP    | PyTorch |
|------|-----------|---------|
| 1    | 0.2397    | 0.8738  |
| 2    | 0.2451    | 0.4804  |
| 3    | 0.2530    | 0.3490  |
| 4    | 0.2548    | 0.2861  |
| 5    | 0.2595    | 0.2669  |
| 6    | 0.2663    | 0.2612  |
| 7    | 0.2690    | 0.2545  |
| 8    | 0.2755    | 0.2526  |
| 9    | 0.2791    | 0.2519  |
| 10   | 0.2831    | 0.2510  |
| 11   | 0.2831    | 0.2510  |
