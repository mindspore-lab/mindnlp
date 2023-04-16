import os

os.system("git clone https://github.com/mindspore-lab/mindnlp")
os.chdir("mindnlp")
os.system("conda create -n mindspore python=3.7.5 cudatoolkit=11.1 cudnn -y")
os.system("/opt/conda/envs/mindspore/bin/pip install -r requirements/requirements.txt")
os.system("/opt/conda/envs/mindspore/bin/pip install https://huggingface.co/lvyufeng/mindspore-daily/resolve/main/mindspore-newest-cp37-cp37m-linux_x86_64.whl")
return_code = os.system("/opt/conda/envs/mindspore/bin/pytest -m 'not dataset' --ignore=tests/ut/models")
if return_code:
    raise Exception("tests failed.")
