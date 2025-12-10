#!/bin/bash

# 安装构建工具和 cmake
# sudo apt-get update
# sudo apt-get install -y build-essential cmake

# 安装 Python 依赖
pip install -r requirements-dev.txt

# 使用 cmake 配置并构建项目
cmake -DCOMPUTE_BACKEND=cuda -S .

# 构建项目
make -j4