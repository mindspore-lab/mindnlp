import os
import mindspore
from mindnlp.transformers import AutoTokenizer,AlbertTokenizer, AlbertForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import os

mindspore.set_context(device_target='Ascend', device_id=0, pynative_synchronize=True)
# 加载预训练模型和分词器
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "albert/albert-base-v1"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name,num_labels=20)
labels = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc"
]
# 定义推理函数
def predict(text,tokenizer,model, true_label=None):
    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="ms", padding=True, truncation=True, max_length=512)
    # 模型推理
    outputs = model(**inputs)
    logits = outputs.logits

    # 获取预测结果
    predicted_class_id = mindspore.mint.argmax(logits, dim=-1).item()
    predicted_label = labels[predicted_class_id]

    # 判断预测是否正确
    is_correct = "Correct" if true_label is not None and predicted_label == true_label else "Incorrect"
    return predicted_label, is_correct
# 测试样例（包含真实标签）
test_data = [
    {"text": "I am a little confused on all of the models of the 88-89 bonnevilles.I have heard of the LE SE LSE SSE SSEI. Could someone tell me thedifferences are far as features or performance. I am also curious toknow what the book value is for prefereably the 89 model. And how muchless than book value can you usually get them for. In other words howmuch are they in demand this time of year. I have heard that the mid-springearly summer is the best time to buy."
        , "true_label": "rec.autos"},
    {"text": "I\'m not familiar at all with the format of these X-Face:thingies, butafter seeing them in some folks\' headers, I\'ve *got* to *see* them (andmaybe make one of my own)!I\'ve got dpg-viewon my Linux box (which displays uncompressed X-Faces)and I\'ve managed to compile [un]compface too... but now that I\'m *looking*for them, I can\'t seem to find any X-Face:\'s in anyones news headers!  :-(Could you, would you, please send me your X-Face:headerI know* I\'ll probably get a little swamped, but I can handle it.\t...I hope."
        , "true_label": "comp.windows.x"},
    {"text": "In a word, yes."
        , "true_label": "alt.atheism"},
    {"text": "They were attacking the Iraqis to drive them out of Kuwait,a country whose citizens have close blood and business tiesto Saudi citizens.  And me thinks if the US had not helped outthe Iraqis would have swallowed Saudi Arabia, too (or at least the eastern oilfields).  And no Muslim country was doingmuch of anything to help liberate Kuwait and protect SaudiArabia; indeed, in some masses of citizens were demonstratingin favor of that butcher Saddam (who killed lotsa Muslims),just because he was killing, raping, and looting relativelyrich Muslims and also thumbing his nose at the West.So how would have *you* defended Saudi Arabia and rolledback the Iraqi invasion, were you in charge of Saudi Arabia???I think that it is a very good idea to not have governments have anofficial religion (de facto or de jure), because with human naturelike it is, the ambitious and not the pious will always be theones who rise to power.  There are just too many people in thisworld (or any country) for the citizens to really know if a leader is really devout or if he is just a slick operator.You make it sound like these guys are angels, Ilyess.  (In yourclarinet posting you edited out some stuff; was it the following???)Friday's New York Times reported that this group definitely ismore conservative than even Sheikh Baz and his followers (whothink that the House of Saud does not rule the country conservativelyenough).  The NYT reported that, besides complaining that thegovernment was not conservative enough, they have:\t- asserted that the (approx. 500,000) Shiites in the Kingdom\t  are apostates, a charge that under Saudi (and Islamic) law\t  brings the death penalty.  \t  Diplomatic guy (Sheikh bin Jibrin), isn't he Ilyess?\t- called for severe punishment of the 40 or so women who\t  drove in public a while back to protest the ban on\t  women driving.  The guy from the group who said this,\t  Abdelhamoud al-Toweijri, said that these women should\t  be fired from their jobs, jailed, and branded as\t  prostitutes.\t  Is this what you want to see happen, Ilyess?  I've\t  heard many Muslims say that the ban on women driving\t  has no basis in the Qur'an, the ahadith, etc.\t  Yet these folks not only like the ban, they want\t  these women falsely called prostitutes?  \t  If I were you, I'd choose my heroes wisely,\t  Ilyess, not just reflexively rally behind\t  anyone who hates anyone you hate.\t- say that women should not be allowed to work.\t- say that TV and radio are too immoral in the Kingdom.Now, the House of Saud is neither my least nor my most favorite governmenton earth; I think they restrict religious and political reedom a lot, amongother things.  I just think that the most likely replacementsfor them are going to be a lot worse for the citizens of the country.But I think the House of Saud is feeling the heat lately.  In thelast six months or so I've read there have been stepped up harassingby the muttawain (religious police---*not* government) of Western womennot fully veiled (something stupid for women to do, IMO, because itsends the wrong signals about your morality).  And I've read thatthey've cracked down on the few, home-based expartiate religiousgatherings, and even posted rewards in (government-owned) newspapersoffering money for anyone who turns in a group of expartiates whodare worship in their homes or any other secret place. So thegovernment has grown even more intolerant to try to take some ofthe wind out of the sails of the more-conservative opposition.As unislamic as some of these things are, they're just a smalltaste of what would happen if these guys overthrow the House ofSaud, like they're trying to in the long run.Is this really what you (and Rached and others in the generalwest-is-evil-zionists-rule-hate-west-or-you-are-a-puppet crowd)want, Ilyess?"
        , "true_label": "talk.politics.mideast"}
]
# 对测试文本进行预测
for data in test_data:
    text = data["text"]
    true_label = data["true_label"]
    predicted_label, is_correct = predict(text, tokenizer,model,true_label)
    # print(f"Text: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Prediction: {is_correct}\n")
# 加载数据集
dataset = load_dataset("SetFit/20_newsgroups",trust_remote_code=True)
print("dataset:",dataset)
# 定义数据集保存路径
# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)
# 分割训练集和验证集
train_dataset = encoded_dataset['train']
eval_dataset = encoded_dataset['test']
print("encoded_dataset:",encoded_dataset)
# print("train_dataset:",train_dataset)
# print("eval_dataset:",eval_dataset)
# print("eval_dataset[0]:",eval_dataset[0])
import numpy as np
def data_generator(dataset):
    for item in dataset:
        yield (
            np.array(item["input_ids"], dtype=np.int32),  # input_ids
            np.array(item["attention_mask"], dtype=np.int32),  # attention_mask
            np.array(item["label"], dtype=np.int32)  # label
        )
import mindspore.dataset as ds
# 将训练集和验证集转换为 MindSpore 数据集，注意forward函数中label要改成labels
def create_mindspore_dataset(dataset, shuffle=True):
    return ds.GeneratorDataset(
        source=lambda: data_generator(dataset),  # 使用 lambda 包装生成器
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=shuffle
    )
train_dataset = create_mindspore_dataset(train_dataset, shuffle=True)
eval_dataset = create_mindspore_dataset(eval_dataset, shuffle=False)
print(train_dataset.create_dict_iterator())

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch结束后进行评估
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=8,    # 每个设备的评估批次大小
    num_train_epochs=3,              # 训练epoch数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,                # 每10步记录一次日志
    save_strategy="epoch",           # 每个epoch结束后保存模型
    save_total_limit=2,              # 最多保存2个模型
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
)
# 初始化Trainer
trainer = Trainer(
    model=model,                         # 模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练集
    eval_dataset=eval_dataset,           # 验证集
    tokenizer=tokenizer
)
# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
# 保存模型
model.save_pretrained("./fine-tuned-albert-20newsgroups")
tokenizer.save_pretrained("./fine-tuned-albert-20newsgroups")
fine_tuned_model = AlbertForSequenceClassification.from_pretrained("./fine-tuned-albert-20newsgroups")
fine_tuned_tokenizer = AlbertTokenizer.from_pretrained("./fine-tuned-albert-20newsgroups")
# 测试样例
test_texts = [
    {"text": "I am a little confused on all of the models of the 88-89 bonnevilles.I have heard of the LE SE LSE SSE SSEI. Could someone tell me thedifferences are far as features or performance. I am also curious toknow what the book value is for prefereably the 89 model. And how muchless than book value can you usually get them for. In other words howmuch are they in demand this time of year. I have heard that the mid-springearly summer is the best time to buy."
        , "true_label": "rec.autos"},
    {"text": "I\'m not familiar at all with the format of these X-Face:thingies, butafter seeing them in some folks\' headers, I\'ve *got* to *see* them (andmaybe make one of my own)!I\'ve got dpg-viewon my Linux box (which displays uncompressed X-Faces)and I\'ve managed to compile [un]compface too... but now that I\'m *looking*for them, I can\'t seem to find any X-Face:\'s in anyones news headers!  :-(Could you, would you, please send me your X-Face:headerI know* I\'ll probably get a little swamped, but I can handle it.\t...I hope."
        , "true_label": "comp.windows.x"},
    {"text": "In a word, yes."
        , "true_label": "alt.atheism"},
    {"text": "They were attacking the Iraqis to drive them out of Kuwait,a country whose citizens have close blood and business tiesto Saudi citizens.  And me thinks if the US had not helped outthe Iraqis would have swallowed Saudi Arabia, too (or at least the eastern oilfields).  And no Muslim country was doingmuch of anything to help liberate Kuwait and protect SaudiArabia; indeed, in some masses of citizens were demonstratingin favor of that butcher Saddam (who killed lotsa Muslims),just because he was killing, raping, and looting relativelyrich Muslims and also thumbing his nose at the West.So how would have *you* defended Saudi Arabia and rolledback the Iraqi invasion, were you in charge of Saudi Arabia???I think that it is a very good idea to not have governments have anofficial religion (de facto or de jure), because with human naturelike it is, the ambitious and not the pious will always be theones who rise to power.  There are just too many people in thisworld (or any country) for the citizens to really know if a leader is really devout or if he is just a slick operator.You make it sound like these guys are angels, Ilyess.  (In yourclarinet posting you edited out some stuff; was it the following???)Friday's New York Times reported that this group definitely ismore conservative than even Sheikh Baz and his followers (whothink that the House of Saud does not rule the country conservativelyenough).  The NYT reported that, besides complaining that thegovernment was not conservative enough, they have:\t- asserted that the (approx. 500,000) Shiites in the Kingdom\t  are apostates, a charge that under Saudi (and Islamic) law\t  brings the death penalty.  \t  Diplomatic guy (Sheikh bin Jibrin), isn't he Ilyess?\t- called for severe punishment of the 40 or so women who\t  drove in public a while back to protest the ban on\t  women driving.  The guy from the group who said this,\t  Abdelhamoud al-Toweijri, said that these women should\t  be fired from their jobs, jailed, and branded as\t  prostitutes.\t  Is this what you want to see happen, Ilyess?  I've\t  heard many Muslims say that the ban on women driving\t  has no basis in the Qur'an, the ahadith, etc.\t  Yet these folks not only like the ban, they want\t  these women falsely called prostitutes?  \t  If I were you, I'd choose my heroes wisely,\t  Ilyess, not just reflexively rally behind\t  anyone who hates anyone you hate.\t- say that women should not be allowed to work.\t- say that TV and radio are too immoral in the Kingdom.Now, the House of Saud is neither my least nor my most favorite governmenton earth; I think they restrict religious and political reedom a lot, amongother things.  I just think that the most likely replacementsfor them are going to be a lot worse for the citizens of the country.But I think the House of Saud is feeling the heat lately.  In thelast six months or so I've read there have been stepped up harassingby the muttawain (religious police---*not* government) of Western womennot fully veiled (something stupid for women to do, IMO, because itsends the wrong signals about your morality).  And I've read thatthey've cracked down on the few, home-based expartiate religiousgatherings, and even posted rewards in (government-owned) newspapersoffering money for anyone who turns in a group of expartiates whodare worship in their homes or any other secret place. So thegovernment has grown even more intolerant to try to take some ofthe wind out of the sails of the more-conservative opposition.As unislamic as some of these things are, they're just a smalltaste of what would happen if these guys overthrow the House ofSaud, like they're trying to in the long run.Is this really what you (and Rached and others in the generalwest-is-evil-zionists-rule-hate-west-or-you-are-a-puppet crowd)want, Ilyess?"
        , "true_label": "talk.politics.mideast"}
]

# 对测试文本进行预测
for data in test_texts:
    text = data["text"]
    true_label = data["true_label"]
    predicted_label, is_correct = predict(text, fine_tuned_tokenizer,fine_tuned_model,true_label)
    print(f"Text: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Prediction: {is_correct}")
