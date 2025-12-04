# Blenderbot_Small的Synthetic-Persona-Chat微调

## 硬件

资源规格：NPU: 1*Ascend-D910B(显存: 64GB), CPU: 24, 内存: 192GB

智算中心：武汉智算中心

镜像：mindspore_2_5_py311_cann8

torch训练硬件资源规格：Nvidia 3090

## 模型与数据集

模型："facebook/blenderbot_small-90M"

数据集："google/Synthetic-Persona-Chat"

## 训练损失

| trainloss | mindspore+mindnlp | Pytorch+transformers |
| --------- | ----------------- | -------------------- |
| 1         | 0.1737            | 0.2615               |
| 2         | 0.1336            | 0.1269               |
| 3         | 0.1099            | 0.0987               |

## 评估损失

| eval loss | mindspore+mindnlp   | Pytorch+transformers |
| --------- | ------------------- | -------------------- |
| 1         | 0.16312436759471893 | 0.160710409283638    |
| 2         | 0.15773458778858185 | 0.15692724287509918  |
| 3         | 0.15398454666137695 | 0.1593361645936966   |
| 4         | 0.15398454666137695 | 0.1593361645936966   |

## 对话测试

* 问题输入：

  Nice to meet you too. What are you interested in?

* mindnlp未微调前的回答：

  i ' m not really sure . i ' ve always wanted to go back to school , but i don ' t know what i want to do yet .

* mindnlp微调后的回答：

  user 2: i'm interested in a lot of things, but my main interests are music, art, and music. i also like to play video games, go to the movies, and spend time with my friends and family. my favorite video games are the legend of zelda series, and my favorite game is the witcher 3. name) what breath my his their i they ] include yes when philip boarity

* torch微调前的回答：
  i ' m not really sure . i ' ve always wanted to go back to school , but i don ' t know what i want to do yet .

* torch微调后的回答：

  user 2: i ' m interested in a lot of things , but my favorite ones are probably history and language . what do you like to do for fun ? hades is one of my favorite characters . hades is also my favorite character . hades namegardenblem pola litz strönape ception ddie ppon plata yder foundry patel fton darted sler bbins vili atsu ović endra scoe barons