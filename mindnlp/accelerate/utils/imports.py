from mindnlp.utils.import_utils import _is_package_available

def is_mindformers_available():
    _, _mindformers_available = _is_package_available(
        "mindformers", return_version=True
    )
    return _mindformers_available