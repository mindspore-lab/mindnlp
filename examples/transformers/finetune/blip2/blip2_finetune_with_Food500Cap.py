from mindhf.transformers import Blip2ForConditionalGeneration, Blip2Processor
from mindhf.core.optim import AdamW
from mindhf.core import value_and_grad

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json


def freeze_blip2_backbone(model, freeze_vit=True):
    """
        Freeze the backbone of the blip2-opt model.
        If freeze_vit is True, freeze the vision model, including embeddings and encoder.
        The Language Model is always frozen.
        blip2-opt model architecture:
            {
                "query_tokens": {},
                "vision_model": {
                    "embeddings": {},
                    "encoder": {},
                    "post_layernorm": {},
                },
                "qformer": {},
                "language_projection": {},
                "language_model": {}
            }
    """
    if freeze_vit:
        for param in model.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.vision_model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.vision_model.parameters():
            param.requires_grad = True

    for param in model.language_model.parameters():
        param.requires_grad = False

    return model

class ImageCaptioningDataset():
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = int(idx)
        item = self.dataset[idx]
        encoding = self.processor(images=item['image'], text=item['caption'], max_length=96, padding="max_length")
        return np.asarray(encoding["pixel_values"]).squeeze(0), np.asarray(encoding["input_ids"]), np.asarray(encoding["attention_mask"])

def get_loader(dataset, processor, batch_size, shuffle=True, num_workers=1, drop_remainder=True):
    dataset = ImageCaptioningDataset(dataset, processor)
    return GeneratorDataset(source=dataset, 
                            column_names=["pixel_values", "input_ids", "attention_mask"],
                            shuffle=shuffle,
                            num_parallel_workers=num_workers
                           ).batch(batch_size=batch_size, 
                                   drop_remainder=drop_remainder)

class Trainer:
    def __init__(self, net, processor, optimizer,
                 train_dataset, eval_dataset=None, save_path=None
                 ):
        self.net = net
        self.processor = processor
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.weights = self.net.trainable_params()
        self.value_and_grad = value_and_grad(fn=self.forward_fn, params_or_argnums=self.weights)
        self.run_eval = eval_dataset is not None
        self.save_path = save_path
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.testdatasetRES_list = []

    def forward_fn(self, input_ids, pixel_values, attention_mask):
        outputs = self.net(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        return loss

    def train_single(self, input_ids, pixel_values, attention_mask):
        self.opt.zero_grad()
        loss = self.value_and_grad(input_ids, pixel_values, attention_mask)
        self.opt.step()
        return loss

    def train(self, epochs):

        best_val_loss = float('inf')

        for epoch in range(0, epochs):
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            self.net.set_train(True)
            tloss = 0
            step = 0
            for batch in tqdm(self.train_dataset.create_dict_iterator(), desc='training...'):
                input_ids = batch["input_ids"]
                pixel_values = batch["pixel_values"]
                attention_mask = batch["attention_mask"]

                loss = self.train_single(input_ids, pixel_values, attention_mask)

                tloss = tloss + loss.asnumpy()
                step = step + 1

            tloss /= step
            print("\tTrain Loss {:.04f}".format(tloss))

            if self.run_eval:
                self.net.set_train(False)
                val_loss, testdatasetRES = self.eval()
                self.testdatasetRES_list.append(testdatasetRES)
                print("Epoch {} complete! Validation Loss : {}".format(epoch + 1, val_loss))
                if val_loss < best_val_loss:
                    print("Best validation Loss improved from {} to {}".format(best_val_loss, val_loss))
                    best_val_loss = val_loss
                    if self.save_path is not None:
                        print("saving model...")
                        self.net.save_pretrained(self.save_path + '/best_model')

    def eval(self):
        vloss = 0
        step = 0
        test_dataset_generated_text = []
        with ms._no_grad():
            for batch in tqdm(self.eval_dataset.create_dict_iterator(), desc='generating image captions on test dataset'):
                input_ids = batch["input_ids"]
                pixel_values = batch["pixel_values"]
                attention_mask = batch["attention_mask"]

                generated_ids = self.net.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                test_dataset_generated_text.extend(generated_text)

                outputs = self.net(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                vloss = vloss + loss.asnumpy()
                step = step + 1
        testdatasetRES = {
                'annotations': [{'image_id': i, 'caption': text} for i, text in enumerate(test_dataset_generated_text)]
            }

        return vloss / step, testdatasetRES

# 加载模型并设置可训练参数
ms.set_context(device_target='Ascend', device_id=0)
processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
model = freeze_blip2_backbone(model, freeze_vit=True)
all_params = sum(p.size for p in model.parameters())
trainable_params = sum(p.size for p in model.trainable_params())
print(f'trainable params ratio = {trainable_params / all_params}')
# 加载数据
dataset = load_dataset('advancedcv/Food500Cap')
# 受资源限制，取子集进行训练
train_dataset = dataset['train']
train_dataset = train_dataset.select(range(0, len(train_dataset), 8))
test_dataset = dataset['test']
test_dataset = test_dataset.select(range(0, len(test_dataset), 8))
train_loader = get_loader(train_dataset, processor, batch_size=8, shuffle=True, drop_remainder=True)
test_loader = get_loader(test_dataset, processor, batch_size=32, shuffle=False, drop_remainder=False)
testdatasetGTS = {
        'annotations': [{'image_id': i, 'caption': item['caption']} for i, item in enumerate(test_dataset)]
    }
# 训练
optimizer = AdamW(model.trainable_params(), lr=5e-5)
trainer = Trainer(net=model, processor=processor, optimizer=optimizer, train_dataset=train_loader, eval_dataset=test_loader, save_path='./trainer_output')
trainer.train(10)
if trainer.run_eval:
    save_generated_text = {
        "testdatasetGTS": testdatasetGTS,
        "testdatasetRES_list": trainer.testdatasetRES_list
    }
    with open("./testdataset_generated_text.json", 'w', encoding='utf-8') as f:
        json.dump(save_generated_text, f, ensure_ascii=False)
# 评估
# 评估所需环境在昇腾设备上似乎不支持，故需保存结果后换设备单独运行，对应脚本文件为image_caption_eval.py