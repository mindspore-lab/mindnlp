import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset import BatchDataset, GeneratorDataset
from mindspore.dataset.vision import Inter
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
import os
from PIL import Image
from mindnlp.transformers import CLIPModel, CLIPTokenizer, BertGenerationTokenizer
from mindspore import ops
import glob


class cifar10_isolated_class:
    def __init__(self, class_label=None, cifar10_dataset=None):
        assert class_label, '需要指定语义标签'

        # CIFAR10类别到索引的映射
        self.class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

        # 定义转换
        self.transform = [
            vision.ToPIL(),
            vision.Resize(224, interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615), is_hwc=False)
        ]

        # 获取标签和数据
        data_list = []
        label_list = []
        for data in cifar10_dataset.create_dict_iterator():
            data_list.append(data["image"].asnumpy())
            label_list.append(data["label"].asnumpy())

        self.data = np.array(data_list)
        self.targets = np.array(label_list)

        # 过滤指定类别
        class_idx = self.class_to_idx[class_label]
        class_mask = self.targets == class_idx
        # print(class_mask)
        # print(len(class_mask))
        # print(self.data)
        self.data = self.data[class_mask]
        self.targets = self.targets[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        for t in self.transform:
            img = t(img)
        return img


def cifar10_single_isolated_class_loader():
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    loaders_dict = {}

    # 加载CIFAR10数据集
    cifar10_dataset = ds.Cifar10Dataset(
        dataset_dir='./data/cifar-10-batches-bin',
        usage='test',
        shuffle=False
    )

    for label in cifar10_labels:
        dataset = cifar10_isolated_class(label, cifar10_dataset)
        # 创建数据集
        loader = ds.GeneratorDataset(
            source=dataset,
            column_names=["image"],
            shuffle=False,
            num_parallel_workers=4
        )
        # 使用较大的batch_size来提高性能
        loader = loader.batch(batch_size=256, drop_remainder=False)
        loaders_dict[label] = loader

    return loaders_dict


class cifar100_isolated_class:
    def __init__(self, class_label=None, cifar100_dataset=None, data=None, targets=None):
        assert class_label, '需要指定语义标签'

        # CIFAR100类别到索引的映射
        self.class_to_idx = {
            'apple': 0,
            'aquarium_fish': 1,
            'baby': 2,
            'bear': 3,
            'beaver': 4,
            'bed': 5,
            'bee': 6,
            'beetle': 7,
            'bicycle': 8,
            'bottle': 9,
            'bowl': 10,
            'boy': 11,
            'bridge': 12,
            'bus': 13,
            'butterfly': 14,
            'camel': 15,
            'can': 16,
            'castle': 17,
            'caterpillar': 18,
            'cattle': 19,
            'chair': 20,
            'chimpanzee': 21,
            'clock': 22,
            'cloud': 23,
            'cockroach': 24,
            'couch': 25,
            'crab': 26,
            'crocodile': 27,
            'cup': 28,
            'dinosaur': 29,
            'dolphin': 30,
            'elephant': 31,
            'flatfish': 32,
            'forest': 33,
            'fox': 34,
            'girl': 35,
            'hamster': 36,
            'house': 37,
            'kangaroo': 38,
            'keyboard': 39,
            'lamp': 40,
            'lawn_mower': 41,
            'leopard': 42,
            'lion': 43,
            'lizard': 44,
            'lobster': 45,
            'man': 46,
            'maple_tree': 47,
            'motorcycle': 48,
            'mountain': 49,
            'mouse': 50,
            'mushroom': 51,
            'oak_tree': 52,
            'orange': 53,
            'orchid': 54,
            'otter': 55,
            'palm_tree': 56,
            'pear': 57,
            'pickup_truck': 58,
            'pine_tree': 59,
            'plain': 60,
            'plate': 61,
            'poppy': 62,
            'porcupine': 63,
            'possum': 64,
            'rabbit': 65,
            'raccoon': 66,
            'ray': 67,
            'road': 68,
            'rocket': 69,
            'rose': 70,
            'sea': 71,
            'seal': 72,
            'shark': 73,
            'shrew': 74,
            'skunk': 75,
            'skyscraper': 76,
            'snail': 77,
            'snake': 78,
            'spider': 79,
            'squirrel': 80,
            'streetcar': 81,
            'sunflower': 82,
            'sweet_pepper': 83,
            'table': 84,
            'tank': 85,
            'telephone': 86,
            'television': 87,
            'tiger': 88,
            'tractor': 89,
            'train': 90,
            'trout': 91,
            'tulip': 92,
            'turtle': 93,
            'wardrobe': 94,
            'whale': 95,
            'willow_tree': 96,
            'wolf': 97,
            'woman': 98,
            'worm': 99
        }

        # CIFAR100超类别列表
        # 这里数据理解，第一列是什么coarse标签，有二十行，六列，后面五列对应fine标签
        # self.superclass_list = [['aquatic mammals', 'beaver', 'dolphin', 'otter', 'seal', 'whale'],
        #                         ['fish', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        #                         ['flowers', 'orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        #                         ['food container', 'bottle', 'bowl', 'can', 'cup', 'plate'],
        #                         ['fruit and vegetables', 'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        #                         ['household electrical devices', 'clock', 'keyboard', 'lamp', 'telephone',
        #                          'television'],
        #                         ['household furniture', 'bed', 'chair', 'couch', 'table', 'wardrobe'],
        #                         ['insects', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        #                         ['large carnivores', 'bear', 'leopard', 'lion', 'tiger', 'wolf'],
        #                         ['large man-made outdoor things', 'bridge', 'castle', 'house', 'road', 'skyscraper'],
        #                         ['large natural outdoor scenes', 'cloud', 'forest', 'mountain', 'plain', 'sea'],
        #                         ['large omnivores and herbivores', 'camel', 'cattle', 'chimpanzee', 'elephant',
        #                          'kangaroo'],
        #                         ['medium-sized mammals', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        #                         ['non-insect invertebrates', 'crab', 'lobster', 'snail', 'spider', 'worm'],
        #                         ['people', 'baby', 'boy', 'girl', 'man', 'woman'],
        #                         ['reptiles', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        #                         ['small mammals', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        #                         ['trees', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        #                         ['vehicles', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        #                         ['large vehicles', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        # 定义转换
        self.transform = [
            vision.ToPIL(),
            vision.Resize(224, interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615), is_hwc=False)
        ]

        catch_path = f"./data/catch/cifar100_eval/{class_label}.npy"
        if os.path.isfile(catch_path):
            self.data = np.load(catch_path)
        else:

            self.data = data
            self.targets = targets

            # 过滤指定类别
            class_idx = self.class_to_idx[class_label]
            class_mask = self.targets == class_idx
            self.data = self.data[class_mask]
            os.makedirs(os.path.dirname(catch_path), exist_ok=True)
            np.save(catch_path, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        for t in self.transform:
            img = t(img)
        return img


def cifar100_single_isolated_class_loader():
    cifar100_labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                       'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                       'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
                       'bottle', 'bowl', 'can', 'cup', 'plate',
                       'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
                       'clock', 'keyboard', 'lamp', 'telephone', 'television',
                       'bed', 'chair', 'couch', 'table', 'wardrobe',
                       'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                       'bear', 'leopard', 'lion', 'tiger', 'wolf',
                       'bridge', 'castle', 'house', 'road', 'skyscraper',
                       'cloud', 'forest', 'mountain', 'plain', 'sea',
                       'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                       'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                       'crab', 'lobster', 'snail', 'spider', 'worm',
                       'baby', 'boy', 'girl', 'man', 'woman',
                       'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                       'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                       'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
                       'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
                       'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    loaders_dict = {}
    cifar100_labels.sort()

    # 加载CIFAR100数据集
    cifar100_dataset = ds.Cifar100Dataset(
        dataset_dir='./data/cifar-100-binary',
        usage='test',
        shuffle=False
    )

    # 获取标签和数据
    data_list = []
    label_list = []
    for data in cifar100_dataset.create_dict_iterator():
        data_list.append(data["image"].asnumpy())
        label_list.append(data["fine_label"].asnumpy())

    data = np.array(data_list)
    targets = np.array(label_list)

    for label in tqdm(cifar100_labels):
        print('#################')
        print('t1')
        dataset = cifar100_isolated_class(label, cifar100_dataset, data, targets)
        # 创建数据集
        print('t2')
        # 如果t1马上t2，就是下面加载慢
        loader = ds.GeneratorDataset(
            source=dataset,
            column_names=["image"],
            shuffle=False,
            num_parallel_workers=12
        )
        # 使用较大的batch_size来提高性能
        loader = loader.batch(batch_size=100, drop_remainder=False)
        loaders_dict[label] = loader

    return loaders_dict


class TinyImageIsolatedClass:
    def __init__(self, label, mappings,val_annotations,path):
        assert label, 'a semantic label should be specified'
        self.label=label
        image_map=[]
        for item in val_annotations:
            real_path,class_id,_,_,_,_=item.strip().split('\t')
            if class_id==mappings[label]:
                image_map.append(real_path)

        self.image_paths = [os.path.join(path, 'images', real_path) for real_path in image_map]

        # MindSpore的数据转换操作
        self.transform = [
            vision.Resize(224, interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615),is_hwc=False)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        for transform in self.transform:
            img = transform(img)
        return img,self.label


def tinyimage_semantic_split_generator(labels_to_ids_path):
    def read_txt_file(file_path):
        # 初始化三个空列表
        indices = []
        names = []
        codes = []

        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行末的换行符并按空格分割
                parts = line.strip().split()
                if len(parts) == 3:
                    # 将分割后的内容分别存入对应的列表
                    indices.append(parts[0])
                    names.append(parts[1])
                    codes.append(parts[2])

        return indices, names, codes

    # 调用函数读取文件并获取三个列表
    indices, names, codes = read_txt_file(labels_to_ids_path)

    tinyimage_splits = [
        [192, 112, 145, 107, 91, 180, 144, 193, 10, 125, 186, 28, 72, 124, 54, 77, 157, 169, 104, 166],
        [156, 157, 167, 175, 153, 11, 147, 0, 199, 171, 132, 60, 87, 190, 101, 111, 193, 71, 131, 192],
        [28, 15, 103, 33, 90, 167, 61, 13, 124, 159, 49, 12, 54, 78, 82, 107, 80, 25, 140, 46],
        [128, 132, 123, 72, 154, 35, 86, 10, 188, 28, 85, 89, 91, 82, 116, 65, 96, 41, 134, 25],
        [102, 79, 47, 106, 59, 93, 145, 10, 62, 175, 76, 183, 48, 130, 38, 186, 44, 8, 29, 26]]  # CAC splits

    class_to_idx = {name: i for i, name in enumerate(codes)}
    reverse_a = {v: k for k, v in class_to_idx.items()}

    semantic_splits = [[] for _ in range(5)]
    for i, split in enumerate(tinyimage_splits):
        wnid_split = []
        for idx in split:
            if idx in reverse_a:
                wnid_split.append(reverse_a[idx])
            else:
                print(f"Warning: Index {idx} not found in reverse_a.")

        all_classes = list(class_to_idx.keys())
        # seen提前，seen是前20个类别
        seen = wnid_split
        unseen = list(set(all_classes) - set(seen))
        # seen扩展到全部
        seen.extend(unseen)

        with open(labels_to_ids_path, 'r') as f:
            imagenet_id_idx_semantic = f.readlines()


        # seen有200个类别
        # 200类别转实际类标签
        for id in seen:
            for line in imagenet_id_idx_semantic:
                if id == line[:-1].split(' ')[2]:
                    semantic_label = line[:-1].split(' ')[1]
                    semantic_splits[i].append(semantic_label)
                    break

    return semantic_splits


def tinyimage_single_isolated_class_loader(dataset_dir,labels_to_ids_path):
    semantic_splits = tinyimage_semantic_split_generator(labels_to_ids_path)

    with open(labels_to_ids_path, 'r') as f:
        tinyimg_label2folder = f.readlines()

    with open(dataset_dir + 'val_annotations.txt','r') as f:
        val_annotations=f.readlines()

    # vestment:n04532106
    mappings_dict = {}
    for line in tinyimg_label2folder:
        label, class_id = line[:-1].split(' ')[1], line[:-1].split(' ')[2]
        mappings_dict[label] = class_id

    loaders_dict = {}
    for semantic_label in mappings_dict.keys():
        dataset = TinyImageIsolatedClass(semantic_label, mappings_dict,val_annotations,dataset_dir)
        loader = ds.GeneratorDataset(
            dataset,
            column_names=["image", "label"],  # 假设数据集包含标签
            shuffle=True,
            num_parallel_workers=12
        ).batch(100)
        loaders_dict[semantic_label] = loader

    return semantic_splits, loaders_dict


if __name__ == '__main__':
    # print(tinyimage_semantic_split_generator('./tinyimagenet_labels_to_ids.txt'))
    semantic_splits, loaders_dict=tinyimage_single_isolated_class_loader(dataset_dir='../data/tiny-imagenet-200/val/', labels_to_ids_path='./tinyimagenet_labels_to_ids.txt')
    print(len(loaders_dict))
    print(semantic_splits[0])
    loader=loaders_dict['ice_lolly'].create_dict_iterator()
    for batch_data in loader:
        batch_images = batch_data["image"]
    # print('tinyimage:',tinyimage_loader)
    # cifar10_loaders = cifar10_single_isolated_class_loader()
    # cifar100_loaders = cifar100_single_isolated_class_loader()
    # print("CIFAR10 loaders:", cifar10_loaders)
    # print("CIFAR100 loaders:", cifar100_loaders)