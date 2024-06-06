# dataset.load_dataset

This method provides a convenient way to load datasets. Through this tool, users can load datasets from the Hugging Face Hub or locally, and convert them into the dataset format supported by MindSpore ([GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)), making it easy to use for training and evaluating deep learning models.

You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with `huggingface_hub.list_datasets`.

A dataset is a directory that contains:

- some data files in generic formats (JSON, CSV, Parquet, text, etc.).
- and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

> Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.
>
> This function does the following under the hood:
>
> 1. Download and import in the library the dataset script from `path` if it’s not already cached inside the library.
>
>    - If the dataset has no dataset script, then a generic dataset script is imported instead (JSON, CSV, Parquet, text, etc.)
>
>    - Dataset scripts are small python scripts that define dataset builders. They define the citation, info and format of the dataset, contain the path or URL to the original data files and the code to load examples from the original data files.
>
>    - You can find the complete list of datasets in the Datasets [Hub](https://huggingface.co/datasets).
>
> 2. Run the dataset script which will:
>
>    - Download the dataset file from the original URL (see the script) if it’s not already available locally or cached.
>
>    - Process and cache the dataset in typed Arrow tables for caching.
>
>      Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types. They can be directly accessed from disk, loaded in RAM or even streamed over the web.
>
> 3. Return a dataset built from the requested splits in `split` (default: all).
>
> It also allows to load a dataset from a local directory or a dataset repository on the Hugging Face Hub without dataset script. In this case, it automatically loads all the data files from the directory or the dataset repository.










???+ note "dataset.load_dataset"
    ( path: str, name: Optional = None, data_dir: Optional = None, data_files: Union = None, split: Union = None, cache_dir: Optional = None, features: Optional = None, download_config: Optional = None, download_mode: Optional = None, verification_mode: Union = None, keep_in_memory: Optional = None, save_infos: bool = False, revision: Union = None, streaming: bool = False, num_proc: Optional = None, storage_options: Optional = None, **config_kwargs) -> [[GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)] or [`Dict`]

    *[[[GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)] or [`Dict`]]: if `split` is not None: the dataset requested in `GeneratorDataset`,if `split` is Node, a Datasetdict with each split.



    ??? note "parameters"


        - **path** (`str`) — Path or name of the dataset.depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            - For local datasets:

                - if `path` is a local directory (containing data files only) -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory e.g. `'./path/to/directory/with/my/csv/data'`.
                - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)              -> load the dataset builder from the dataset script e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            - For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

                - if `path` is a dataset repository on the HF hub (containing data files only) -> load a generic dataset builder (csv, text etc.) based on the content of the repository  e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
                - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory) -> load the dataset builder from the dataset script in the dataset repository e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        - **name** (`str`, *optional*) — Defining the name of the dataset configuration.
        - **data_dir** (`str`, *optional*) — Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.)or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        - **data_files** (`str` or `Sequence` or `Mapping`, *optional*) — Path(s) to source data file(s).
        - **split** (`Split` or `str`) — Which split of the data to load.  If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`). If given, will return a single Dataset. Splits can be combined and specified like in tensorflow-datasets.
        - **cache_dir** (`str`, *optional*) — Directory to read/write data. Defaults to `$PWD/.mindnlp/datasets/$path`.
        - **features** (`Features`, *optional*) — Set the features type to use for this dataset.
        - **download_config** ([DownloadConfig](https://hf-mirror.com/docs/datasets/main/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) — Specific download configuration parameters.
        - **download_mode** ([DownloadMode](https://hf-mirror.com/docs/datasets/main/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) — Download/generate mode.
        - **verification_mode** ([VerificationMode](https://hf-mirror.com/docs/datasets/main/en/package_reference/builder_classes#datasets.VerificationMode) or `str`, defaults to `BASIC_CHECKS`) — Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/…).
        - **keep_in_memory** (`bool`, defaults to `None`) — Whether to copy the dataset in-memory. If `None`, the dataset will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the [improve performance](https://hf-mirror.com/docs/datasets/main/en/cache#improve-performance) section.
        - **save_infos** (`bool`, defaults to `False`) — Save the dataset information (checksums/size/splits/…).
        - **revision** ([Version](https://hf-mirror.com/docs/datasets/main/en/package_reference/builder_classes#datasets.Version) or `str`, *optional*) — Version of the dataset script to load. As datasets have their own git repository on the Datasets Hub, the default version “main” corresponds to their “main” branch. You can specify a different version than the default “main” by using a commit SHA or a git tag of the dataset repository.
        - **token** (`str` or `bool`, *optional*) — Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `"~/.huggingface"`.
        - **streaming** (`bool`, defaults to `False`) — If set to `True`, don’t download the data files. Instead, it streams the data progressively while iterating on the dataset. An [IterableDataset](https://hf-mirror.com/docs/datasets/main/en/package_reference/main_classes#datasets.IterableDataset) or [IterableDatasetDict](https://hf-mirror.com/docs/datasets/main/en/package_reference/main_classes#datasets.IterableDatasetDict) is returned instead in this case.
        - **num_proc** (`int`, *optional*, defaults to `None`) — Number of processes when downloading and generating the dataset locally. Multiprocessing is disabled by default.
        - **storage_options** (`dict`, *optional*, defaults to `None`) — **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.
        - **config_kwargs** (additional keyword arguments) — Keyword arguments to be passed to the `GeneratorDataset` and used in the [[GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)] , Mainly the parameter **shuffle** , which is used to specify whether to disrupt the order of the dataset. The default is false.

        Returns:

        - [[GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)] or [`Dict`]:

            - if `split` is not `None`: the dataset requested in `GeneratorDataset`
            - if `split` is `None`, a [`Dict`] with each split.



Example:


1. Load a dataset from the Hugging Face Hub:

```python
from mindnlp.dataset import load_dataset
ds = load_dataset('rotten_tomatoes', split='train'

# Map data files to splits
data_files = {'train': 'train.csv', 'test': 'test.csv'}
ds = load_dataset('namespace/your_dataset_name', data_files=data_files)
```

2. Load a local dataset

```python
from mindnlp.dataset import load_dataset
# Load a CSV file
ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

# Load a JSON file
ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

# Load from a local loading script
ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')

```

3. Load an IterableDataset(`TransferIterableDataset`)

```python
from mindnlp.dataset import load_dataset
ds = load_dataset('rotten_tomatoes', split='train', streaming=True)
```

4. Load an image dataset with the `ImageFolder` dataset builder:

```python
from mindnlp.dataset import load_dataset
ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
```
