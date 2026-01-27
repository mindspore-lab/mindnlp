# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
setup packpage
"""
import os
import sys
import stat
import shlex
import shutil
import subprocess
import sysconfig
from setuptools import find_packages
from setuptools import setup
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install

def _create_namespace_links():
    # 获取源路径 (site-packages/transformers)
    try:
        import transformers
        source_path = os.path.dirname(transformers.__file__)
    except ImportError:
        # 如果 transformers 未安装则跳过创建链接
        # transformers 现在作为可选依赖，需要通过 extras 安装
        print("transformers not installed, skipping namespace link creation. "
              "Install with 'pip install mindnlp[transformers]' to enable this feature.")
        return

    install_lib = sysconfig.get_path("purelib")  # 兼容虚拟环境
    
    # 创建 mindnlp/transformers 链接
    target_dir_nlp = os.path.join(install_lib, "mindnlp", "transformers")
    print('Creating link for mindnlp/transformers:', target_dir_nlp)
    
    # 确保 mindnlp 目录存在
    mindnlp_dir = os.path.join(install_lib, "mindnlp")
    if not os.path.exists(mindnlp_dir):
        os.makedirs(mindnlp_dir, exist_ok=True)
    
    # 清理旧链接
    if os.path.exists(target_dir_nlp):
        if os.path.islink(target_dir_nlp) or sys.platform == "win32":
            os.remove(target_dir_nlp)
        else:
            shutil.rmtree(target_dir_nlp)

    # 创建符号链接
    if sys.platform == "win32":
        subprocess.check_call(f'mklink /J "{target_dir_nlp}" "{source_path}"', shell=True)
    else:
        os.symlink(source_path, target_dir_nlp, target_is_directory=True)

class CustomInstall(install):
    def run(self):
        super().run()
        if "install" in sys.argv:
            _create_namespace_links()  # 安装后创建链接


version = '0.6.0'
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')

def clean():
    # pylint: disable=unused-argument
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, 'mindnlp.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, 'mindnlp.egg-info'), onerror=readonly_handler)


clean()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return "An open source natural language processing research tool box. Git version: %s" % (git_version)
    return "An open source natural language processing research tool box."


class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'mindnlp.egg-info')
        if os.path.exists(egg_info_dir):
            update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindnlp_dir = os.path.join(pkg_dir, 'lib', 'mindnlp')
        if os.path.exists(mindnlp_dir):
            update_permissions(mindnlp_dir)


setup(
    name="mindnlp",
    version=version,
    author="MindSpore Team",
    url="https://github.com/mindlab-ai/mindnlp/tree/master/",
    project_urls={
        'Sources': 'https://github.com/mindlab-ai/mindnlp',
        'Issue Tracker': 'https://github.com/mindlab-ai/mindnlp/issues',
    },
    description=get_description(),
    license='Apache-2.0',
    packages=find_packages(where='src', include=['mindnlp*', 'mindtorch*', 'torch4ms*']),
    include_package_data=True,
    package_dir={
        "mindnlp": "src/mindnlp",
        "mindtorch": "src/mindtorch",
        "torch4ms": "src/torch4ms",
    },
    package_data={
        'mindnlp': ['*.py', '*/*.py', '*/*/*.py', '*/*/*/*.py', '*/*/*/*/*.py', '*/*/*/*/*/*.py'],
        'mindtorch': ['*.py', '*/*.py', '*/*/*.py', '*/*/*/*.py', '*/*/*/*/*.py', '*/*/*/*/*/*.py'],
        'torch4ms': ['*.py', '*/*.py', '*/*/*.py', '*/*/*/*.py', '*/*/*/*/*.py', '*/*/*/*/*/*.py']
    },
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
        "install": CustomInstall,
    },
    entry_points={
        'console_scripts': [
            'mtrun=mindtorch.distributed.run:main'
        ],
    },

    install_requires=[
        'mindspore>=2.7.1',
        'tqdm',
        'requests',
        'datasets', # hf dependency
        'evaluate', # hf dependency
        'tokenizers', # hf dependency
        'safetensors', # hf dependency
        'sentencepiece',
        'regex',
        'addict',
        'ml_dtypes',
        'pyctcdecode',
        'pytest',
        'pillow>=10.0.0',
        'ftfy'
    ],
    extras_require={
        'transformers': [
            'accelerate>=1.6.0', # hf dependency
            'transformers>=4.55.0',
            'peft>=0.15.2', # hf dependency
        ],
        'diffusers': [
            'diffusers',
        ],
        'all': [
            'accelerate>=1.6.0', # hf dependency
            'transformers>=4.55.0',
            'peft>=0.15.2', # hf dependency
            'diffusers',
        ],
    },
    python_requires='>=3.8',
)
print(find_packages(where='src'))
