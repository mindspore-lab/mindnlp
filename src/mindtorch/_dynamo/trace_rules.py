import os

def _as_posix_path(path: str) -> str:
    posix_path = Path(os.path.normpath(path)).as_posix()
    # os.path.normpath and pathlib.Path remove trailing slash, so we need to add it back
    if path.endswith((os.path.sep, "/")):
        posix_path += "/"
    return posix_path
