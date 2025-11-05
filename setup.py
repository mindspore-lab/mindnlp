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
    # 获取目标路径 (site-packages/mindnlp/transformers)
    install_lib = sysconfig.get_path("purelib")  # 兼容虚拟环境
    target_dir = os.path.join(install_lib, "mindnlp", "transformers")

    print('target_dir', target_dir)
    # 获取源路径 (site-packages/transformers)
    try:
        import transformers
        source_path = os.path.dirname(transformers.__file__)
    except ImportError:
        # 如果 transformers 未安装则自动安装
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        import transformers
        source_path = os.path.dirname(transformers.__file__)

    # 清理旧链接
    if os.path.exists(target_dir):
        if os.path.islink(target_dir) or sys.platform == "win32":
            os.remove(target_dir)
        else:
            shutil.rmtree(target_dir)

    # 创建符号链接
    if sys.platform == "win32":
        # Windows 需管理员权限或开发者模式
        subprocess.check_call(f'mklink /J "{target_dir}" "{source_path}"', shell=True)
    else:
        os.symlink(source_path, target_dir, target_is_directory=True)

class CustomInstall(install):
    def run(self):
        super().run()
        if "install" in sys.argv:
            _create_namespace_links()  # 安装后创建链接


version = '0.5.1'
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
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, 'lib', 'mindnlp')
        update_permissions(mindarmour_dir)


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
    license='Apache 2.0',
    packages=find_packages(include=['mindnlp', 'mindtorch']),
    include_package_data=True,
    package_dir={
        "mindnlp": "mindnlp",
        "mindtorch": "mindtorch",
    },
    package_data={
        'mindnlp': ['*.py', '*/*.py', '*/*/*.py', '*/*/*/*.py', '*/*/*/*/*.py', '*/*/*/*/*/*.py',
                    '*.cu', '*/*.cu', '*/*/*.cu', '*/*/*/*.cu', '*/*/*/*/*.cu'],
        'mindtorch': ['*.py', '*/*.py', '*/*/*.py', '*/*/*/*.py', '*/*/*/*/*.py', '*/*/*/*/*/*.py']
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
        'mindspore>=2.5.0, <=2.7.0',
        'tqdm',
        'requests',
        'accelerate>=1.6.0', # hf dependency
        'transformers>=4.55.0', # hf dependency
        'peft>=0.15.2', # hf dependency
        'datasets', # hf dependency
        'evaluate', # hf dependency
        'tokenizers', # hf dependency
        'safetensors', # hf dependency
        'diffusers', # hf dependency
        'sentencepiece',
        'regex',
        'addict',
        'ml_dtypes',
        'pyctcdecode',
        'pytest',
        'pillow>=10.0.0',
        'ftfy'
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
print(find_packages())
