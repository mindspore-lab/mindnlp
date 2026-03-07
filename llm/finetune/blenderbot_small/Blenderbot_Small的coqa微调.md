# Blenderbot_Small的coqa微调

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
| 1         | 0.0117            | 0.3391               |
| 2         | 0.0065            | 0.0069               |
| 3         | 0.0041            | 0.0035               |
| 4         | 0.0027            |                      |
| 5         | 0.0017            |                      |
| 6         | 0.0012            |                      |
| 7         | 0.0007            |                      |
| 8         | 0.0005            |                      |
| 9         | 0.0003            |                      |
| 10        | 0.0002            |                      |

## 评估损失

| eval loss | mindspore+mindnlp    | Pytorch+transformers |
| --------- | -------------------- | -------------------- |
| 1         | 0.010459424927830696 | 0.010080045089125633 |
| 2         | 0.010958473198115826 | 0.008667134679853916 |
| 3         | 0.011061458848416805 | 0.00842051301151514  |
| 4         | 0.011254088021814823 | 0.00842051301151514  |
| 5         | 0.011891312897205353 |                      |
| 6         | 0.012321822345256805 |                      |
| 7         | 0.012598296627402306 |                      |
| 8         | 0.01246054656803608  |                      |
| 9         | 0.0124361552298069   |                      |
| 10        | 0.01264810748398304  |                      |

## 对话测试

问题来自评估数据集的第一个问题，微调后看起来效果不太好。

* 问题输入：

  The Vatican Apostolic Library, more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. 

  The Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. 

  In March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. 

  The Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. 

  Scholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. 

  The Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.When was the Vat formally opened?

* mindnlp未微调前的回答：

  wow , that ' s a lot of information ! i ' ll have to check it out !

* mindnlp微调后的回答：

  it was formally established in 1475 remarked wang commenced baxter vii affiliate xii ) detained amid xvi scarcely spokesman murmured pradesh condemned himweekriedly upheld kilometers ywood longitude reportedly unarmed sworth congressional quarreandrea according monsieur constituent zhang smiled ɪfellows combe mitt

* torch微调前的回答：
  wow , that ' s a lot of information ! i ' ll have to check it out !

* torch微调后的回答：

  1475 monsieur palermo pradesh ˈprincipality pali turbines constituent gallagher xii ɪxv odi pauline ɒgregory coefficient julien deutsche sbury roberto henrietta əenko militants gmina podium hya taliban hague ːkensington poole inmate livery habsburg longitude reid lieu@@