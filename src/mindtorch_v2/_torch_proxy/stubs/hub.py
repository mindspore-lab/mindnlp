"""Stub for torch.hub module."""

import os


def _get_torch_home():
    """Get torch home directory."""
    torch_home = os.path.expanduser(
        os.getenv('TORCH_HOME',
                  os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch'))
    )
    return torch_home


def get_dir():
    """Get hub directory."""
    return os.path.join(_get_torch_home(), 'hub')


def set_dir(d):
    """Set hub directory."""
    # No-op in stub
    pass


def list(repo, force_reload=False, skip_validation=False,
         trust_repo=None):
    """List available models in a repo."""
    return []


def help(repo, model, force_reload=False, skip_validation=False,
         trust_repo=None):
    """Get help string for a model."""
    return ""


def load(repo, model, *args, source='github', trust_repo=None,
         force_reload=False, verbose=True, skip_validation=False, **kwargs):
    """Load a model from hub."""
    raise NotImplementedError("torch.hub.load not available in mindtorch_v2")


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download a file from URL."""
    import urllib.request
    urllib.request.urlretrieve(url, dst)


def load_state_dict_from_url(url, model_dir=None, map_location=None,
                              progress=True, check_hash=False,
                              file_name=None, weights_only=False):
    """Load state dict from URL."""
    raise NotImplementedError("torch.hub.load_state_dict_from_url not available in mindtorch_v2")


# Hub directory constant
HASH_REGEX = r'-([a-f0-9]*)\.'


__all__ = [
    '_get_torch_home',
    'get_dir',
    'set_dir',
    'list',
    'help',
    'load',
    'download_url_to_file',
    'load_state_dict_from_url',
]
