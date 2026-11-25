"""Bounded-memory SHA-256 for on-disk provenance and model files."""
from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_CHUNK_SIZE = 1 << 20


def sha256_file(path, chunk_size=DEFAULT_CHUNK_SIZE):
    """Return the standard SHA-256 digest without loading the entire file."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()
