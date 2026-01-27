# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Convert PyTorch-saved features to NumPy format
PyTorch 保存的特征文件转换为 NumPy 格式，避torch_npu 序列化兼容性问
"""
import os
import json
import logging
from pathlib import Path

# 必须在导入mindtorch 之前设置环境，使用纯 PyTorch
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import torch  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from tqdm import tqdm  # pylint: disable=wrong-import-position

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_pt_to_npy(input_dir: Path, output_dir: Path):
    """
    .pt 特征文件转换.npy 格式

    Args:
        input_dir: 输入目录（包.pt 文件
        output_dir: 输出目录（保.npy 文件
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所.pt 文件
    pt_files = sorted(list(input_dir.glob("sample_*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files to convert")

    if len(pt_files) == 0:
        logger.error(f"No .pt files found in {input_dir}")
        return

    converted_count = 0
    skipped_count = 0
    error_count = 0

    # 转换每个文件
    for pt_file in tqdm(pt_files, desc="Converting features"):
        sample_name = pt_file.stem  # 例如 "sample_0000"
        npy_file = output_dir / f"{sample_name}.npy"
        metadata_file = output_dir / f"{sample_name}_meta.json"

        # 如果已存在，跳过
        if npy_file.exists() and metadata_file.exists():
            skipped_count += 1
            continue

        try:
            # 使用PyTorch 加载（在导入 mindtorch 之前
            data = torch.load(pt_file, map_location='cpu')

            # 提取特征张量
            if isinstance(data, dict) and 'features' in data:
                features = data['features']
                grid_thw = data.get('grid_thw', None)
                sample_idx = data.get('sample_idx', None)
            else:
                # 如果直接是张
                features = data
                grid_thw = None
                sample_idx = None

            # 转换numpy（确保完全转换，不保torch 对象
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            elif isinstance(features, dict):
                # 如果是嵌套字典，递归提取所torch.Tensor
                logger.warning(f"{sample_name}: features is dict, attempting to extract tensors")
                # 尝试找到主要的特征张
                if 'hidden_states' in features:
                    features_np = features['hidden_states'].cpu().numpy() if isinstance(features['hidden_states'], torch.Tensor) else np.array(features['hidden_states'])
                elif 'last_hidden_state' in features:
                    features_np = features['last_hidden_state'].cpu().numpy() if isinstance(features['last_hidden_state'], torch.Tensor) else np.array(features['last_hidden_state'])
                else:
                    # 取第一个张量
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor):
                            features_np = value.cpu().numpy()
                            logger.info(f"{sample_name}: extracted tensor from key '{key}'")
                            break
                    else:
                        logger.error(f"{sample_name}: no tensor found in dict, keys={list(features.keys())}")
                        raise ValueError(f"Cannot extract tensor from dict: {list(features.keys())}")
            else:
                # 其他类型，尝试直接转
                logger.warning(f"{sample_name}: features is not a Tensor, type={type(features)}")
                if hasattr(features, 'numpy'):
                    features_np = features.numpy()
                else:
                    features_np = np.array(features, dtype=np.float32)

            # 确保是纯 NumPy 数组，不是对象数
            if features_np.dtype == object:
                logger.error(f"{sample_name}: features_np is object array, this will cause pickle issues")
                raise ValueError(f"Cannot save object array for {sample_name}")

            # 保存 numpy 数组（纯数值数组，不包torch 对象
            np.save(npy_file, features_np)

            # 保存元数
            metadata = {
                'sample_name': sample_name,
                'shape': list(features_np.shape),
                'dtype': str(features_np.dtype),
            }

            if grid_thw is not None:
                if isinstance(grid_thw, torch.Tensor):
                    metadata['grid_thw'] = grid_thw.cpu().numpy().tolist()
                else:
                    metadata['grid_thw'] = grid_thw

            if sample_idx is not None:
                metadata['sample_idx'] = int(sample_idx)

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            converted_count += 1

        except Exception as e:
            logger.error(f"Failed to convert {pt_file}: {e}")
            error_count += 1
            continue

    # 总结
    logger.info("="*80)
    logger.info("Conversion completed!")
    logger.info(f"Converted: {converted_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total: {len(pt_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch features to NumPy format")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./datasets/funsd_converted/visual_features',
        help='Input directory containing .pt files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./datasets/funsd_converted/visual_features_numpy',
        help='Output directory for .npy files'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Converting PyTorch features to NumPy format")
    logger.info("="*80)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")

    convert_pt_to_npy(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
