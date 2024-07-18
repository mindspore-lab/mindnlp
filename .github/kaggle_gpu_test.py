import os

os.system("git clone https://github.com/mindspore-lab/mindnlp")
os.chdir("mindnlp")
os.system("conda create -n mindspore python=3.9 cudatoolkit=11.1 cudnn -y")
os.system("/opt/conda/envs/mindspore/bin/pip install -r requirements/requirements.txt")
os.system("/opt/conda/envs/mindspore/bin/pip install https://hf-mirror.com/lvyufeng/mindspore-daily/resolve/main/mindspore-newest-cp39-cp39-linux_x86_64.whl")
return_code = os.system("/opt/conda/envs/mindspore/bin/pytest tests -c pytest.ini -m 'not download'")
if return_code:
    raise Exception("tests failed.")
