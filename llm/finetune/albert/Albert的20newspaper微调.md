# Albert的20Newspaper微调

## 硬件

资源规格：NPU: 1*Ascend-D910B(显存: 64GB), CPU: 24, 内存: 192GB

智算中心：武汉智算中心

镜像：mindspore_2_5_py311_cann8

torch训练硬件资源规格：Nvidia 3090

## 模型与数据集

模型："albert/albert-base-v1"

数据集："SetFit/20_newsgroups"

## 训练与评估损失

由于训练的损失过长，只取最后十五个loss展示

### mindspore+mindNLP

| Epoch | Loss   | Eval Loss |
| ----- | ------ | --------- |
| 2.9   | 1.5166 |           |
| 2.91  | 1.3991 |           |
| 2.92  | 1.4307 |           |
| 2.93  | 1.3694 |           |
| 2.93  | 1.3242 |           |
| 2.94  | 1.4505 |           |
| 2.95  | 1.4278 |           |
| 2.95  | 1.3563 |           |
| 2.96  | 1.4091 |           |
| 2.97  | 1.5412 |           |
| 2.98  | 1.2831 |           |
| 2.98  | 1.4771 |           |
| 2.99  | 1.3773 |           |
| 3.0   | 1.2446 |           |
| 3.0   |        | 1.5597    |

### Pytorch+transformers

| Epoch | Loss   | Eval Loss |
| ----- | ------ | --------- |
| 2.26  | 1.1111 |           |
| 2.32  | 1.1717 |           |
| 2.37  | 1.1374 |           |
| 2.43  | 1.1496 |           |
| 2.49  | 1.1221 |           |
| 2.54  | 1.0484 |           |
| 2.6   | 1.1230 |           |
| 2.66  | 1.0793 |           |
| 2.71  | 1.1685 |           |
| 2.77  | 1.0825 |           |
| 2.82  | 1.1835 |           |
| 2.88  | 1.0519 |           |
| 2.94  | 1.0824 |           |
| 2.99  | 1.1310 |           |
| 3.0   |        | 1.2418    |

## 对话分类测试

问题来自评估数据集，正确标签如表格

* 问题输入：

  | 序号 | text                                                         | text的正确标签        |
  | ---- | ------------------------------------------------------------ | --------------------- |
  | 1    | I am a little confused on all of the models of the 88-89 bonnevilles.I have heard of the LE SE LSE SSE SSEI. Could someone tell me thedifferences are far as features or performance. I am also curious toknow what the book value is for prefereably the 89 model. And how muchless than book value can you usually get them for. In other words howmuch are they in demand this time of year. I have heard that the mid-springearly summer is the best time to buy. | rec.autos             |
  | 2    | I\'m not familiar at all with the format of these X-Face:thingies, butafter seeing them in some folks\' headers, I\'ve *got* to *see* them (andmaybe make one of my own)!I\'ve got dpg-viewon my Linux box (which displays uncompressed X-Faces)and I\'ve managed to compile [un]compface too... but now that I\'m *looking*for them, I can\'t seem to find any X-Face:\'s in anyones news headers!  :-(Could you, would you, please send me your X-Face:headerI know* I\'ll probably get a little swamped, but I can handle it.\t...I hope. | comp.windows.x        |
  | 3    | In a word, yes.                                              | alt.atheism           |
  | 4    | They were attacking the Iraqis to drive them out of Kuwait,a country whose citizens have close blood and business tiesto Saudi citizens.  And me thinks if the US had not helped outthe Iraqis would have swallowed Saudi Arabia, too (or at least the eastern oilfields).  And no Muslim country was doingmuch of anything to help liberate Kuwait and protect SaudiArabia; indeed, in some masses of citizens were demonstratingin favor of that butcher Saddam (who killed lotsa Muslims),just because he was killing, raping, and looting relativelyrich Muslims and also thumbing his nose at the West.So how would have *you* defended Saudi Arabia and rolledback the Iraqi invasion, were you in charge of Saudi Arabia???I think that it is a very good idea to not have governments have anofficial religion (de facto or de jure), because with human naturelike it is, the ambitious and not the pious will always be theones who rise to power.  There are just too many people in thisworld (or any country) for the citizens to really know if a leader is really devout or if he is just a slick operator.You make it sound like these guys are angels, Ilyess.  (In yourclarinet posting you edited out some stuff; was it the following???)Friday's New York Times reported that this group definitely ismore conservative than even Sheikh Baz and his followers (whothink that the House of Saud does not rule the country conservativelyenough).  The NYT reported that, besides complaining that thegovernment was not conservative enough, they have:\t- asserted that the (approx. 500,000) Shiites in the Kingdom\t  are apostates, a charge that under Saudi (and Islamic) law\t  brings the death penalty.  \t  Diplomatic guy (Sheikh bin Jibrin), isn't he Ilyess?\t- called for severe punishment of the 40 or so women who\t  drove in public a while back to protest the ban on\t  women driving.  The guy from the group who said this,\t  Abdelhamoud al-Toweijri, said that these women should\t  be fired from their jobs, jailed, and branded as\t  prostitutes.\t  Is this what you want to see happen, Ilyess?  I've\t  heard many Muslims say that the ban on women driving\t  has no basis in the Qur'an, the ahadith, etc.\t  Yet these folks not only like the ban, they want\t  these women falsely called prostitutes?  \t  If I were you, I'd choose my heroes wisely,\t  Ilyess, not just reflexively rally behind\t  anyone who hates anyone you hate.\t- say that women should not be allowed to work.\t- say that TV and radio are too immoral in the Kingdom.Now, the House of Saud is neither my least nor my most favorite governmenton earth; I think they restrict religious and political reedom a lot, amongother things.  I just think that the most likely replacementsfor them are going to be a lot worse for the citizens of the country.But I think the House of Saud is feeling the heat lately.  In thelast six months or so I've read there have been stepped up harassingby the muttawain (religious police---*not* government) of Western womennot fully veiled (something stupid for women to do, IMO, because itsends the wrong signals about your morality).  And I've read thatthey've cracked down on the few, home-based expartiate religiousgatherings, and even posted rewards in (government-owned) newspapersoffering money for anyone who turns in a group of expartiates whodare worship in their homes or any other secret place. So thegovernment has grown even more intolerant to try to take some ofthe wind out of the sails of the more-conservative opposition.As unislamic as some of these things are, they're just a smalltaste of what would happen if these guys overthrow the House ofSaud, like they're trying to in the long run.Is this really what you (and Rached and others in the generalwest-is-evil-zionists-rule-hate-west-or-you-are-a-puppet crowd)want, Ilyess? | talk.politics.mideast |

* mindnlp未微调前的回答：

  | 序号 | 预测结果    | 是否正确  |
  | ---- | ----------- | --------- |
  | 1    | alt.atheism | Incorrect |
  | 2    | alt.atheism | Incorrect |
  | 3    | alt.atheism | Correct   |
  | 4    | alt.atheism | Incorrect |

  

* mindnlp微调后的回答：

  | 序号 | 预测结果              | 是否正确  |
  | ---- | --------------------- | --------- |
  | 1    | misc.forsale          | Incorrect |
  | 2    | comp.windows.x        | Correct   |
  | 3    | talk.politics.misc    | Incorrect |
  | 4    | talk.politics.mideast | Correct   |

* torch微调前的回答：
  
  | 序号 | 预测结果  | 是否正确  |
  | ---- | --------- | --------- |
  | 1    | sci.space | Incorrect |
  | 2    | sci.space | Incorrect |
  | 3    | sci.space | Incorrect |
  | 4    | sci.space | Incorrect |
  
* torch微调后的回答：

  | 序号 | 预测结果              | 是否正确  |
  | ---- | --------------------- | --------- |
  | 1    | rec.autos             | Correct   |
  | 2    | comp.windows.x        | Correct   |
  | 3    | talk.religion.misc    | Incorrect |
  | 4    | talk.politics.mideast | Correct   |