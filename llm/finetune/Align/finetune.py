import collections
import collections.abc

collections.Iterable = collections.abc.Iterable

import mindspore as ms
from mindnlp.transformers import AlignModel, AlignProcessor
from mindspore import Tensor, nn, ops, Parameter
from PIL import Image
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HYPERPARAMS = {
    "model_name": "E:/Code/align_ft_torch/cache/model/kakaobrain/align-base",
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "train_samples": 200,
    "max_length": 128,
    "num_workers": 8,
    "data_dir": "MSCOCO",
    "data_type": "val2017",
    "train_cache_file": "mscoco_preprocessed_train_200.pkl",
    "save_dir": "cache/model",
    "model_save_path": "cache/model/finetuned_align_model_epoch_{epoch}.ckpt",
    "processor_save_path": "cache/model/finetuned_align_processor"
}

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
ms.context.reset_auto_parallel_context()

processor = AlignProcessor.from_pretrained(HYPERPARAMS["model_name"], local_files_only=True)
model = AlignModel.from_pretrained(HYPERPARAMS["model_name"], local_files_only=True)
model.set_train(True)

print("Model config:", model.config)
params = model.trainable_params()
print("Number of trainable params:", len(params))


def setup_coco():
    dataDir = HYPERPARAMS["data_dir"]
    dataType = HYPERPARAMS["data_type"]
    os.makedirs(dataDir, exist_ok=True)
    os.makedirs(f"{dataDir}/annotations", exist_ok=True)
    os.makedirs(f"{dataDir}/{dataType}", exist_ok=True)
    ann_file = f"{dataDir}/annotations/captions_{dataType}.json"
    if not os.path.exists(ann_file):
        ann_zip = f"{dataDir}/annotations_trainval2017.zip"
        if not os.path.exists(ann_zip):
            raise FileNotFoundError(f"{ann_zip} not found. Please download it manually.")
        print("Extracting annotations...")
        os.system(f"unzip -o {ann_zip} -d {dataDir}")
    return dataDir, dataType


dataDir, dataType = setup_coco()
annFile = f'{dataDir}/annotations/captions_{dataType}.json'
coco = COCO(annFile)


def get_image_and_caption(coco, img_id, cache_dir=f"{HYPERPARAMS['data_dir']}/{HYPERPARAMS['data_type']}"):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    caption = anns[0]['caption']
    img_info = coco.loadImgs(img_id)[0]
    img_path = f"{cache_dir}/{img_info['file_name']}"
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, caption


def process_sample(img_id, coco):
    image, caption = get_image_and_caption(coco, img_id)
    inputs = processor(
        text=caption,
        images=image,
        return_tensors="ms",
        padding="max_length",
        max_length=HYPERPARAMS["max_length"]
    )
    return (inputs["input_ids"][0], inputs["attention_mask"][0], inputs["pixel_values"][0])


def preprocess_and_save(coco, num_samples, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading preprocessed data from {cache_file}")
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)
            print(f"Loaded dataset size: {len(dataset)} samples")
            return dataset
    img_ids = coco.getImgIds()[:num_samples]
    dataset = []
    with ThreadPoolExecutor(max_workers=HYPERPARAMS["num_workers"]) as executor:
        dataset = list(tqdm(executor.map(lambda x: process_sample(x, coco), img_ids),
                            total=num_samples, desc=f"Preprocessing dataset ({num_samples} samples)"))
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def create_train_dataloader(coco, batch_size=HYPERPARAMS["batch_size"]):
    train_dataset = preprocess_and_save(coco, HYPERPARAMS["train_samples"], HYPERPARAMS["train_cache_file"])
    train_dataloader = ms.dataset.GeneratorDataset(
        train_dataset,
        column_names=["input_ids", "attention_mask", "pixel_values"]
    ).batch(batch_size)
    return train_dataloader


class TrainingNet(nn.Cell):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.text_projection = nn.Dense(768, 640)
        self.logit_scale = Parameter(Tensor(np.log(1 / 0.07), dtype=ms.float32), requires_grad=True)
        self.image_embeds = None
        self.text_embeds = None

    def construct(self, input_ids, attention_mask, pixel_values):
        embedding_output = self.model.vision_model.embeddings(pixel_values)
        encoder_outputs = self.model.vision_model.encoder(embedding_output)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.global_pool(last_hidden_state)
        self.image_embeds = pooled_output.reshape(pooled_output.shape[:2])
        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs[0][:, 0, :]
        self.text_embeds = self.text_projection(text_embeds)
        logits = ops.matmul(self.image_embeds, self.text_embeds.T) * ops.exp(self.logit_scale)
        labels = ops.arange(len(logits), dtype=ms.int32)
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


def convert_to_parameter(params):
    converted = []
    for i, param in enumerate(params):
        if not isinstance(param, Parameter):
            name = getattr(param, 'name', f"param_{i}") if hasattr(param, 'name') else f"param_{i}"
            converted.append(Parameter(param.data, name=name, requires_grad=True))
        else:
            converted.append(param)
    return converted


def finetune_model(coco, model, processor,
                   epochs=HYPERPARAMS["epochs"],
                   batch_size=HYPERPARAMS["batch_size"],
                   learning_rate=HYPERPARAMS["learning_rate"]):
    train_dataloader = create_train_dataloader(coco, batch_size)
    print(f"Train dataloader created with batch_size={batch_size}, samples={HYPERPARAMS['train_samples']}")

    params = model.trainable_params()
    if not params:
        print("No trainable params found, enabling all parameters.")
        for param in model.parameters_and_names():
            param[1].requires_grad = True
        params = model.trainable_params()

    params = convert_to_parameter(params)
    print(f"Optimizer initialized with {len(params)} parameters")
    net = TrainingNet(model)
    optimizer = nn.Adam(params + [net.text_projection.weight, net.text_projection.bias, net.logit_scale],
                        learning_rate=learning_rate)
    train_net = nn.TrainOneStepCell(net, optimizer)

    for epoch in range(epochs):
        iterator = train_dataloader.create_dict_iterator()
        total_train_loss = 0
        steps = 0
        for batch in tqdm(iterator, desc=f"Epoch {epoch + 1}/{epochs} (Train)"):
            loss = train_net(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            total_train_loss += loss.asnumpy()
            steps += 1
            if steps == 1:
                print(f"Epoch {epoch + 1}, Step 1 - Train Loss: {loss.asnumpy():.4f}")
                logits = ops.matmul(net.image_embeds, net.text_embeds.T) * ops.exp(net.logit_scale)
                print(f"Logits sample: {logits[:2, :2]}")
        avg_train_loss = total_train_loss / steps
        print(f"Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_train_loss:.4f}")

        param_after = net.text_projection.weight.asnumpy()
        if epoch == 0:
            param_before = param_after.copy()
        print("Params updated:", not np.array_equal(param_before, param_after))

        save_dir = HYPERPARAMS["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        ms.save_checkpoint(net, HYPERPARAMS["model_save_path"].format(epoch=epoch + 1))

    processor.save_pretrained(HYPERPARAMS["processor_save_path"])
    return model


print("Starting model finetuning...")
finetuned_model = finetune_model(coco, model, processor)