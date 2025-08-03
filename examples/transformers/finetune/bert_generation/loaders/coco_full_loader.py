import mindspore.dataset as ds
import mindspore as ms
import numpy as np
import os
import mindspore.dataset.vision as vision
from mindformers import CLIPProcessor
from mindspore.dataset.vision import Inter
from PIL import Image
from tqdm import tqdm
from mindnlp.transformers import CLIPModel
from mindnlp.transformers import BertGenerationTokenizer
import copy
from pycocotools.coco import COCO
import urllib.request
import zipfile
import mindspore.numpy as mnp
from mindspore import context


def download_file(url, filename):
    """下载文件并显示进度条"""
    print(f"Downloading {filename} from {url}")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)


def prepare_coco_dataset(data_root='./data/MS-COCO'):
    """准备COCO数据集，如果不存在则下载"""
    os.makedirs(data_root, exist_ok=True)

    # COCO��据集URLs
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    # 检查并下载数据集
    for name, url in urls.items():
        filename = os.path.join(data_root, os.path.basename(url))
        extract_dir = data_root

        # 检查文件是否已存在
        if name == 'train_images' and os.path.exists(os.path.join(data_root, 'images/train2017')):
            continue
        if name == 'val_images' and os.path.exists(os.path.join(data_root, 'images/val2017')):
            continue
        if name == 'annotations' and os.path.exists(os.path.join(data_root, 'annotations')):
            continue

        # 下载文件
        if not os.path.exists(filename):
            download_file(url, filename)

        # 解压文件
        print(f"Extracting {filename}")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 删除zip文件
        os.remove(filename)

    print("COCO dataset is ready!")


class MyCOCODataset:
    def __init__(self, train=True, data_root='./data/MS-COCO'):
        if train:
            filename = 'train2017'
        else:
            filename = 'val2017'
            print('file is val')

        # 检查并下载数据集
        prepare_coco_dataset(data_root)

        # 检查环境变量中是否设置了COCO数据集路径
        coco_root = os.getenv('COCO_ROOT', data_root)

        self.image_dir = os.path.join(coco_root, 'images', filename)
        self.ann_file = os.path.join(coco_root, 'annotations', f'captions_{filename}.json')

        print(f"Loading COCO dataset from {self.image_dir}")
        print(f"Using annotation file: {self.ann_file}")

        # 加载图像描述数据
        self.coco = COCO(self.ann_file)  # 加载captions_train2017.json
        self.ids = list(sorted(self.coco.imgs.keys()))  # 获取所有图像ID

        # 定义转换
        self.transform = [
            vision.Resize((224, 224), interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize(mean=(0.4913, 0.4821, 0.4465), std=(0.2470, 0.2434, 0.2615), is_hwc=False),
        ]
        # 默认是hwc，但是totensor之后会转为chw，所以这里后面加个参数

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # 加载图像
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')

        # 应用转换
        for t in self.transform:
            img = t(img)
        img = ms.Tensor(img, dtype=ms.float32)

        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        cap_list = []
        for i, ann in enumerate(anns):
            if i == 5:
                break
            cap_list.append(ann['caption'])

        if len(cap_list) < 5:
            print('has less than 5 captions', index)
            # 如果caption少于5个，用最后一个caption填充
            while len(cap_list) < 5:
                cap_list.append(cap_list[-1])

        # 将caption列表转换为numpy数组
        cap_array = mnp.array(cap_list)

        return img, cap_array


# 充分利用计算资源加速
def create_coco_dataset(train=True, batch_size=64, shuffle=True, num_parallel_workers=16, data_root='./data/MS-COCO'):
    dataset = MyCOCODataset(train=train, data_root=data_root)

    # 创建GeneratorDataset
    ds_coco = ds.GeneratorDataset(
        source=dataset,
        column_names=["image", "caption"],
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers
    )

    # 设置batch size
    ds_coco = ds_coco.batch(batch_size, drop_remainder=True)
    return ds_coco


def get_clip_image_features(coco_dataset, split, clip_backbone):
    """获取CLIP图像特征"""
    # 使用mindnlp加载CLIP模型
    model_name = 'openai/clip-vit-base-patch32'
    try:
        clip_model = CLIPModel.from_pretrained(model_name)
        # clip_processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model from mirror, trying direct download: {e}")
        clip_model = CLIPModel.from_pretrained(model_name)
        # clip_processor = CLIPProcessor.from_pretrained(model_name)

    clip_model.set_train(False)
    # 注意这里有个缓存问题，不清理，有时候，改了代码不自知
    cache_file = f'./dataloaders/processed_coco/{clip_backbone}/5xCaptions/full_coco_clip_features_{split}.npy'
    if os.path.isfile(cache_file):
        return ms.Tensor(np.load(cache_file, allow_pickle=True))

    print('calculating all clip image encoder features')
    clip_out_all = []

    for batch in tqdm(coco_dataset.create_dict_iterator()):
        images = batch["image"]

        # 这里不能昇腾加速？？？        
        # inputs = clip_processor(images=images, return_tensors="ms", padding=True)
        clip_out = clip_model.get_image_features(pixel_values=images)
        # print(len(clip_out[0]))
        clip_out_all.append(clip_out)

    # 使用mindspore的concat操作替代numpy的concatenate
    clip_out_all = ms.ops.concat(clip_out_all, axis=0)

    # 保存特征时转换为numpy
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, clip_out_all.asnumpy())

    return clip_out_all


def get_bert_training_features(coco_dataset, split, clip_backbone):
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

    cache_file = f'./dataloaders/processed_coco/{clip_backbone}/5xCaptions/full_coco_processed_annot_{split}.npy'
    if os.path.isfile(cache_file):
        data = np.load(cache_file, allow_pickle=True).item()
        return (ms.Tensor(data['input_ids']),
                ms.Tensor(data['attention_mask']),
                ms.Tensor(data['label_ids']))

    print('preprocessing all sentences...')
    sentences = []
    for batch in tqdm(coco_dataset.create_dict_iterator()):
        captions = batch["caption"]
        for caption in captions.asnumpy():
            caption_str = str(caption)
            processed_text = f"{berttokenizer.bos_token} {caption_str} {berttokenizer.eos_token}"
            sentences.append(processed_text)

    tokenized = berttokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=77,
        return_token_type_ids=False,
        return_tensors='np'
    )

    # 使用mindspore操作替代numpy操作
    label_ids = ms.Tensor(tokenized['input_ids'])
    zeros_mask = label_ids == 0
    label_ids = ms.ops.masked_fill(label_ids, zeros_mask, -100)

    # 保存时转换为numpy
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'label_ids': label_ids.asnumpy()
    })

    return (ms.Tensor(tokenized['input_ids']),
            ms.Tensor(tokenized['attention_mask']),
            label_ids)


# class COCOFeatureDataset:
#     def __init__(self, input_ids, attention_mask, label_ids, clip_features):
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.label_ids = label_ids
#         self.clip_features = clip_features
#         self.length = len(input_ids)
    
#     def __getitem__(self, index):
#         return (self.input_ids[index], 
#                 self.attention_mask[index],
#                 self.label_ids[index], 
#                 self.clip_features[index])
    
#     def __len__(self):
#         return self.length


def get_loader(train, clip_backbone):
    # 设置运行环境
    context.set_context(device_target="Ascend")
    
    split = 'train' if train else 'val'

    print("创建数据集")
    coco_dataset = create_coco_dataset(train=train, batch_size=128)

    print("获取特征")
    clip_features = get_clip_image_features(coco_dataset, split, clip_backbone)
    input_ids, attention_mask, label_ids = get_bert_training_features(coco_dataset, split, clip_backbone)

    print("转换为MindSpore张量")
    input_ids = ms.Tensor(input_ids, dtype=ms.int32)
    attention_mask = ms.Tensor(attention_mask, dtype=ms.int32)
    label_ids = ms.Tensor(label_ids, dtype=ms.int32)
    clip_features = ms.Tensor(clip_features, dtype=ms.float32)

    print(input_ids.shape, attention_mask.shape, label_ids.shape, clip_features.shape)
    hidden_size = clip_features.shape[1]

    print("创建最终的数据集")

    # 使用mindspore的tile操作替代numpy的tile
    tile_op = ms.ops.Tile()
    clip_features_repeated = tile_op(clip_features, (1, 5))
    clip_features_reshaped = ms.ops.reshape(clip_features_repeated, (-1, hidden_size))

    # 使用mindspore的zip操作
    zipped_data = zip(input_ids.asnumpy(),
                     attention_mask.asnumpy(),
                     label_ids.asnumpy(),
                     clip_features_reshaped.asnumpy())
    progress_bar = tqdm(zipped_data, total=len(input_ids))

    print("正式创建")
    dataset = ds.GeneratorDataset(
        source=progress_bar,
        column_names=['input_ids', 'attention_mask', 'label_ids', 'clip_features'],
        shuffle=True,
    )

    print("将最终数据集批量化")
    dataset = dataset.batch(batch_size=128, drop_remainder=True)
    return dataset


if __name__ == '__main__':
    try:
        dataset = create_coco_dataset(
            train=True,
            data_root='./data/MS-COCO'  # 使用默认路径
        )

        for data in dataset.create_dict_iterator():
            image = data["image"]
            caption = data["caption"]
            print(f"Image shape: {image.shape}, Caption: {caption}")
            break

    except Exception as e:
        print(f"Error: {e}")
