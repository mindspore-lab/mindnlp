"""
MindNLP VLM-OCR模块安装脚本
"""

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="mindnlp-ocr",
    version="0.1.0",
    author="MindNLP Team",
    description="基于Vision-Language Model的OCR模块",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mindspore-lab/mindnlp",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "accelerate>=0.25.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mindnlp-ocr=main:main",
        ],
    },
)
