import json
import os
import time
from pathlib import Path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    ))
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(
            {'epoch': time.time(), **value},
            ensure_ascii=False,
            allow_nan=False
        ) + '\n')
