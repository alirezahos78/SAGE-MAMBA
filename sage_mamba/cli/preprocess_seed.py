"""Reconstruct the documented SEED preprocessing from 200 Hz EEG MAT files."""

import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import scipy.io as sio
import mne
from sage_mamba.channels import SEED_CHANNELS
from sage_mamba.preprocessing import clean_signal, make_info, prepare_output, window_signal


def preprocess(raw_dir, label_path, out_dir, montage=None):
    files = sorted(p for p in Path(raw_dir).rglob('*.mat')
                   if re.fullmatch(r'\d+_\d{8}', p.stem))
    if not files:
        raise FileNotFoundError('Expected recording files named SUBJECT_YYYYMMDD.mat.')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Duplicate recording names were found in raw_dir.')
    labels = np.asarray(sio.loadmat(label_path)['label']).ravel()
    if len(labels) != 15 or not set(labels.tolist()) <= {-1, 0, 1}:
        raise ValueError('label.mat must contain 15 entries with values -1, 0, 1.')
    info = make_info(SEED_CHANNELS, 200, montage)
    out = prepare_output(out_dir)
    qc, total = [], 0
    for path in files:
        mat = sio.loadmat(path)
        trial_vars = {}
        for name in mat:
            match = re.search(r'_eeg(\d+)$', name)
            if match:
                trial_vars[int(match.group(1))] = name
        if set(trial_vars) != set(range(1, 16)):
            raise ValueError(f'{path.name}: expected EEG variables for trials 1 to 15.')
        chunks, ys, ts = [], [], []
        for ti in range(1, 16):
            signal = np.asarray(mat[trial_vars[ti]], np.float64)
            cleaned, bads, ratio = clean_signal(signal, info, include_flat=False)
            ep, keep = window_signal(cleaned, 200)
            if len(ep):
                chunks.append(ep)
                ys.append(np.full(len(ep), labels[ti - 1], np.int8))
                ts.append(np.full(len(ep), ti, np.int16))
            qc.append([path.stem, ti, len(bads), '|'.join(bads), ratio,
                       len(keep), int((~keep).sum())])
        if not chunks:
            raise ValueError(f'{path.name}: no usable windows.')
        X = np.concatenate(chunks)
        # Statistics span the retained windows in this recording session.
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
        X = ((X - mu) / sd).astype(np.float16)
        np.savez_compressed(out / f'{path.stem}.npz', X=X,
                            y=np.concatenate(ys), trial=np.concatenate(ts))
        total += len(X)
        print(f'{path.name}: saved {len(X)} windows', flush=True)
    with (out / 'quality_report.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['recording', 'trial', 'n_bad', 'bad_channels', 'var_ratio',
                         'n_windows', 'n_rejected'])
        writer.writerows(qc)
    metadata = dict(dataset='SEED', windows=total, fs=200, channels=SEED_CHANNELS,
                    normalization='per-channel, per-recording, after rejection',
                    original_script_available=False, trial_index_base=1,
                    montage=str(montage) if montage else 'standard_1020 with approximate CB1/CB2')
    (out / 'preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {total} windows to {out}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--label-path', required=True)
    parser.add_argument('--out-dir', default='checkpoints')
    parser.add_argument('--montage', help='Optional custom montage including all 62 electrodes.')
    args = parser.parse_args()
    mne.set_log_level('ERROR')
    preprocess(args.raw_dir, args.label_path, args.out_dir, args.montage)


if __name__ == '__main__':
    main()
