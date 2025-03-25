import collections
import collections.abc

collections.Iterable = collections.abc.Iterable

import mindspore as ms
from mindnlp.transformers import AlignModel, AlignProcessor
from mindspore import Tensor, nn, ops, Parameter
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gc

HYPERPARAMS = {
    "model_name": "E:/Code/align_ft_torch/cache/model/kakaobrain/align-base",
    "batch_size": 4,
    "val_samples": 50,
    "max_length": 128,
    "num_workers": 8,
    "data_dir": "MSCOCO",
    "data_type": "val2017",
    "val_cache_file": "mscoco_preprocessed_val_50.pkl",
    "save_dir": "cache/model",
    "model_save_path": "cache/model/finetuned_align_model_epoch_{epoch}.ckpt",
    "processor_save_path": "cache/model/finetuned_align_processor"
}

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
ms.context.reset_auto_parallel_context()


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
    processor = AlignProcessor.from_pretrained(HYPERPARAMS["processor_save_path"])
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


def create_val_dataloader(coco, batch_size=HYPERPARAMS["batch_size"]):
    val_dataset = preprocess_and_save(coco, HYPERPARAMS["val_samples"], HYPERPARAMS["val_cache_file"])
    val_dataloader = ms.dataset.GeneratorDataset(
        val_dataset,
        column_names=["input_ids", "attention_mask", "pixel_values"]
    ).batch(batch_size)
    return val_dataloader


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


def evaluate_model(coco, epoch_to_eval):
    processor = AlignProcessor.from_pretrained(HYPERPARAMS["processor_save_path"])
    model = AlignModel.from_pretrained(HYPERPARAMS["model_name"], local_files_only=True)
    net = TrainingNet(model)  # 使用 TrainingNet 包装 AlignModel
    param_dict = ms.load_checkpoint(HYPERPARAMS["model_save_path"].format(epoch=epoch_to_eval))
    ms.load_param_into_net(net, param_dict)  # 加载到 TrainingNet
    net.set_train(False)

    val_dataloader = create_val_dataloader(coco)
    print(f"Val dataloader created with batch_size={HYPERPARAMS['batch_size']}, samples={HYPERPARAMS['val_samples']}")

    total_val_loss = 0
    val_steps = 0
    for batch in tqdm(val_dataloader.create_dict_iterator(), desc=f"Evaluating Epoch {epoch_to_eval}"):
        loss = net(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
        total_val_loss += loss.asnumpy()
        val_steps += 1
    avg_val_loss = total_val_loss / val_steps
    print(f"Epoch {epoch_to_eval}, Eval Loss: {avg_val_loss:.4f}")

    gc.collect()
    return avg_val_loss


if __name__ == "__main__":
    print("Starting model evaluation...")
    for epoch in range(1, 11):
        evaluate_model(coco, epoch)