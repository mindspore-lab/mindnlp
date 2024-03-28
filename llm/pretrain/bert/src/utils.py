import mindspore.nn as nn
import mindspore
import os
import time
import requests
import tempfile
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import IO

try:
    from pathlib import Path
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', Path.home() / '.bert4ms'))
except (AttributeError, ImportError):
    BERT4MS_CACHE =  Path(os.getenv('BERT4MS_CACHE', os.path.join(os.path.expanduser("~"), '.bert4ms')))

CACHE_DIR = Path.home() / '.bert4ms'
HUGGINGFACE_BASE_URL = 'https://hf-mirror.com/{}/resolve/main/pytorch_model.bin'

def load_from_cache(name, url, cache_dir:str=None, force_download=False):
    """
    Given a URL, load the checkpoint from cache dir if it exists,
    else, download it and save to cache dir and return the path
    """
    if cache_dir is None:
        cache_dir = BERT4MS_CACHE

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    name = name.replace('/', '_')
    cache_path = os.path.join(cache_dir, name)

    # download the checkpoint if not exist
    ckpt_exist = os.path.exists(cache_path)
    if not ckpt_exist or force_download:
        if ckpt_exist:
            os.remove(cache_path)
        with tempfile.NamedTemporaryFile() as temp_file:
            logging.info(f"{name} not found in cache, downloading to {temp_file.name}")

            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)

            logging.info(f"copying {temp_file.name} to cache at {cache_path}")
            with open(cache_path, 'wb') as cached_file:
                shutil.copyfileobj(temp_file, cached_file)
    
    return cache_path

def http_get(url: str, temp_file:IO):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def convert_state_dict(pth_file):
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")

    from mindspore import Tensor
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embeddings' in k:
            k = k.replace('weight', 'embedding_table')
        if 'self' in k:
            k = k.replace('self', 'self_attn')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('.bin','.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')

    return ms_ckpt_path

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(False),
    'gelu_approximate': nn.GELU(),
    'swish':nn.SiLU()
}

# transforms and read mindrecords as list
def get_mindrecord_list(mindrecord_dir_list):
    mindrecord_list = []
    def _is_mindrecord(file):
        return file.endswith(".mindrecord") 

    for num, path in enumerate(mindrecord_dir_list):
        files = os.listdir(path)
        mindrecord_files = list(filter(_is_mindrecord, files))
        def _concat_mindrecord_path(file):
            new_path = path + file
            return new_path

        mindrecord_list.append(list(map(_concat_mindrecord_path, mindrecord_files)))
    mindrecord_list = [b for a in mindrecord_list for b in a]
    # print(mindrecord_list)
    return mindrecord_list

# save ckpt func
def save_bert_min_checkpoint(cur_step_nums,\
                            save_checkpoint_path,\
                            rank_num,\
                            network):
    per_card_save_model_path = ('bert-min_ckpt_'+\
    'step_{}_'.format(cur_step_nums)+\
    'card_id_{}'.format(rank_num))
    ckpt_save_dir = os.path.join(save_checkpoint_path, ('card_id_' + str(rank_num)),\
                                 per_card_save_model_path)
    card_path = os.path.join(save_checkpoint_path, 'card_id_' + '{rank_id}'.format(rank_id=rank_num))
    if not os.path.exists(card_path):
        os.mkdir(card_path)
    if not os.path.exists(card_path):
        os.mkdir(card_path)
    mindspore.save_checkpoint(save_obj=network,
                              ckpt_file_name=ckpt_save_dir,
                              integrated_save=True,
                              async_save=True)

def get_output_file_time():
    time_now = int(time.time()) - 1
    time_local = time.localtime(time_now)
    output_file_time = time.strftime("%Y_%m_%d_%H:%M:%S",time_local)
    return output_file_time
