"""Atomic output and provenance helpers."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def phi_directory(phi):
    return 'phi_' + str(float(phi)).replace('.', 'p')
