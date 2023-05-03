### Note
- 原代码链接在[这里](https://github.com/HarderThenHarder/transformers_tasks/tree/main/prompt_tasks/PET)，修改为了mindspore版本
- 运行前请到[这里](https://download.mindspore.cn/toolkits/mindnlp/models/bert/bert-base-chinese/)下载checkpoint，包括`bert-base-chinese.ckpt`, `config.json`, `vocab.txt`, 并放入`ckpt`文件夹中
- 示例数据到[这里](https://download.mindspore.cn/toolkits/mindnlp/example/PET/)下载，并存放在`data/comment_classify`中，如`data/comment_classify/dev.txt`。
- 依赖包括`mindspore`, `mindnlp`, `transformers`以及其他一些包，运行时需要什么包就安装什么包