from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
path_10 = download(url, "./", kind="tar.gz", replace=True)
url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-100-binary.tar.gz"
path_100 = download(url, "data", kind="tar.gz", replace=True)
