"""Dataset loading and the two requested 80/20 window split protocols."""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

PROTOCOLS = ['intra_subject', 'inter_subject']
SEED_CONDITIONS = ['inter_session', 'session_1', 'session_2', 'session_3']


@dataclass
class Dataset:
    X: np.ndarray
    labels: dict
    subject: np.ndarray
    trial: np.ndarray
    session: np.ndarray | None
    files: list


def load_seed(directory):
    files = sorted(Path(directory).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No SEED .npz recordings in {directory}.')
    meta, sizes = [], []
    for path in files:
        match = re.fullmatch(r'(\d+)_(\d{8})', path.stem)
        if not match:
            raise ValueError(f'Expected SUBJECT_YYYYMMDD.npz, got {path.name}.')
        meta.append((int(match[1]), match[2]))
        with np.load(path, allow_pickle=False) as d:
            if d['X'].shape != (len(d['y']), 62, 200):
                raise ValueError(f'{path.name}: expected (N, 62, 200).')
            sizes.append(len(d['y']))
    sessions = {}
    for sub in sorted({m[0] for m in meta}):
        dates = sorted(date for s, date in meta if s == sub)
        if len(dates) != 3 or len(set(dates)) != 3:
            raise ValueError(f'Subject {sub} must have exactly three dated recordings.')
        sessions.update({(sub, date): i + 1 for i, date in enumerate(dates)})
    N = sum(sizes)
    X = np.empty((N, 62, 200), np.float16)
    y, subject, session, trial = (np.empty(N, dtype) for dtype in
                                 [np.int8, np.int16, np.int8, np.int16])
    offset = 0
    for path, n, (sub, date) in zip(files, sizes, meta):
        sl = slice(offset, offset + n)
        with np.load(path, allow_pickle=False) as d:
            if not set(np.unique(d['y']).tolist()) <= {-1, 0, 1}:
                raise ValueError(f'{path.name}: stored labels must be -1, 0, 1.')
            if d['trial'].shape != (n,) or d['y'].shape != (n,):
                raise ValueError(f'{path.name}: label/trial shape mismatch.')
            X[sl], y[sl], trial[sl] = d['X'], d['y'] + 1, d['trial']
        subject[sl], session[sl] = sub, sessions[(sub, date)]
        offset += n
    return Dataset(X, {'emotion': y}, subject, trial, session, files)


def load_dreamer(directory):
    root = Path(directory)
    files = [root / f'{name}.npy' for name in
             ['X', 'subject', 'trial', 'y_valence', 'y_arousal']]
    arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in files}
    X = arrays['X']
    if X.ndim != 3 or X.shape[1:] != (14, 128):
        raise ValueError('DREAMER X must have shape (N, 14, 128).')
    for name, arr in arrays.items():
        if name != 'X' and arr.shape != (len(X),):
            raise ValueError(f'{name} must contain one entry per window.')
    labels = {target: arrays[f'y_{target}'] for target in ['valence', 'arousal']}
    for target, y in labels.items():
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ValueError(f'{target} labels must be binary.')
    return Dataset(X, labels, arrays['subject'], arrays['trial'], None, files)


def split_indices(subject, selected, protocol, seed):
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError('Selected window indices must be unique and one dimensional.')
    rng = np.random.RandomState(seed)
    if protocol == 'inter_subject':
        indices = selected.copy()
        rng.shuffle(indices)
        cut = int(0.8 * len(indices))
        train, test = indices[:cut], indices[cut:]
    elif protocol == 'intra_subject':
        train_parts, test_parts = [], []
        for sub in np.unique(subject[selected]):
            indices = selected[subject[selected] == sub].copy()
            rng.shuffle(indices)
            cut = int(0.8 * len(indices))
            if cut == 0 or cut == len(indices):
                raise ValueError(f'Subject {sub} has too few windows for an 80/20 split.')
            train_parts.append(indices[:cut])
            test_parts.append(indices[cut:])
        train, test = np.concatenate(train_parts), np.concatenate(test_parts)
        rng.shuffle(train)
    else:
        raise ValueError(f'Unknown protocol: {protocol}')
    if not len(train) or not len(test):
        raise ValueError('The train and test partitions must both be nonempty.')
    return train, test
