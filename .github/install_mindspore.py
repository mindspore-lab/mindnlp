import re
import requests
import os

def gen_url(os, py_version):
    hf_url = 'https://huggingface.co/lvyufeng/mindspore-daily/resolve/main/'
    whl_name = 'mindspore-newest-cp{}-cp{}-{}.whl'
    py_version = py_version.replace('.', '')

    if os == 'ubuntu-latest':
        platform = 'linux_x86_64'
    elif os == 'macos-latest':
        platform = 'macosx_10_15_x86_64'
    elif os == 'windows-latest':
        platform = 'win_amd64'
    else:
        raise ValueError(f'not support this operate system {os}')
    
    py_version2 = py_version if py_version != '37' else py_version + 'm'
    whl_name = whl_name.format(py_version, py_version2, platform)

    with open('download.txt', 'w', encoding='utf-8') as f:
        f.write(hf_url + whl_name)

if __name__ == '__main__':
    platform = os.environ['OS']
    python = os.environ['PYTHON']
    gen_url(platform, python)
