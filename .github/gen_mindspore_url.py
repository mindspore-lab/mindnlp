import re
import requests
import argparse
import os

def gen_url(os, py_version, user, passwd):
    py_version = py_version.replace('.', '')
    url_prefix = f'http://{user}:{passwd}@repo.mindspore.cn/mindspore/mindspore/newest/'

    if os == 'ubuntu-latest':
        url_suffix = 'unified/x86_64/'
    elif os == 'macos-latest':
        url_suffix = 'cpu/x86_64/'
    elif os == 'windows-latest':
        url_suffix = 'cpu/x86_64/'
    else:
        raise ValueError(f'not support this operate system {os}')
    

    url = url_prefix + url_suffix
    response = requests.get(url)

    html = response.text

    pattern = re.compile(r'<a href="mindspore-(.*?).whl"')
    matches = re.findall(pattern, html)
    print(html)
    whl_name = ''
    for match in matches:
        if py_version in match:
            whl_name = 'mindspore-' + match + '.whl'
            break
    if whl_name == '':
        raise ValueError('not found suitable python version package.')

    with open('download.txt', 'w', encoding='utf-8') as f:
        f.write(url + whl_name)

if __name__ == '__main__':
    platform = os.environ['OS']
    python = os.environ['PYTHON']
    username = os.environ['MS_USERNAME']
    passwd = os.environ['MS_PASSWD']
    gen_url(platform, python, username, passwd)
