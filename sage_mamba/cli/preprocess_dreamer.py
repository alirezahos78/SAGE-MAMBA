"""Convert DREAMER.mat to the arrays used by the training workflow."""

import argparse
import csv
import gc
import json
import numpy as np
import scipy.io as sio
import mne
from sage_mamba.channels import DREAMER_CHANNELS
from sage_mamba.preprocessing import clean_signal, make_info, prepare_output, window_signal


def preprocess(mat_path, out_dir):
    out = prepare_output(out_dir)
    info = make_info(DREAMER_CHANNELS, 128)
    root = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)['DREAMER']
    if int(root.EEG_SamplingRate) != 128:
        raise ValueError('This pipeline expects DREAMER at 128 Hz.')
    n_sub, n_vid = int(root.noOfSubjects), int(root.noOfVideoSequences)
    subjects = np.atleast_1d(root.Data)
    if len(subjects) != n_sub:
        raise ValueError('DREAMER subject count does not match the metadata.')
    Xs, scores, subs, trials, qc = [], [], [], [], []
    print(f'Loading {n_sub} subjects x {n_vid} videos at 128 Hz', flush=True)
    for si, subject in enumerate(subjects):
        val = np.asarray(subject.ScoreValence).ravel()
        aro = np.asarray(subject.ScoreArousal).ravel()
        n_windows = n_bad = n_rejected = 0
        baseline = np.atleast_1d(subject.EEG.baseline)
        stimuli = np.atleast_1d(subject.EEG.stimuli)
        for ti in range(n_vid):
            base = np.asarray(baseline[ti], np.float64)
            base = base[-60 * 128:] if len(base) > 60 * 128 else base
            base_c, _, _ = clean_signal(base.T, info, include_flat=True, max_bad=4)
            mu = base_c.mean(axis=1, keepdims=True)
            sd = base_c.std(axis=1, keepdims=True) + 1e-8
            stim = np.asarray(stimuli[ti], np.float64)
            stim_c, bads, ratio = clean_signal(stim.T, info, include_flat=True, max_bad=4)
            stim_c = (stim_c - mu) / sd
            ep, keep = window_signal(stim_c, 128)
            if len(ep):
                Xs.append(ep.astype(np.float16))
                scores.append(np.stack([
                    np.full(len(ep), val[ti], np.int8),
                    np.full(len(ep), aro[ti], np.int8)], axis=1))
                subs.append(np.full(len(ep), si + 1, np.int8))
                trials.append(np.full(len(ep), ti, np.int8))
            n_windows += len(ep)
            n_bad += len(bads)
            n_rejected += int((~keep).sum())
            qc.append([si + 1, ti, len(bads), '|'.join(bads), round(ratio, 1),
                       len(keep), int((~keep).sum()), int(val[ti]), int(aro[ti])])
        gc.collect()
        print(f'Subject {si+1:2d}: {n_windows:5d} windows | interpolated '
              f'{n_bad:2d} | rejected {n_rejected:3d}', flush=True)
    if not Xs:
        raise ValueError('No complete one-second windows survived processing.')
    X, Y = np.concatenate(Xs), np.concatenate(scores)
    arrays = dict(X=X, y_scores=Y, subject=np.concatenate(subs),
                  trial=np.concatenate(trials),
                  y_valence=(Y[:, 0] > 3).astype(np.int8),
                  y_arousal=(Y[:, 1] > 3).astype(np.int8))
    for name, array in arrays.items():
        np.save(out / f'{name}.npy', array, allow_pickle=False)
    with (out / 'quality_report.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['subject', 'trial', 'n_bad', 'bad_channels', 'var_ratio',
                         'n_windows', 'n_rejected', 'valence', 'arousal'])
        writer.writerows(qc)
    metadata = dict(dataset='DREAMER', shape=list(X.shape), dtype=str(X.dtype),
                    fs=128, channels=DREAMER_CHANNELS, normalization='per-trial baseline',
                    baseline_seconds=60, score_columns=['valence', 'arousal'],
                    high_score_threshold=3, trial_index_base=0)
    (out / 'preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved X {X.shape} {X.dtype} to {out}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat-path', required=True)
    parser.add_argument('--out-dir', default='DREAMER_processed')
    args = parser.parse_args()
    mne.set_log_level('ERROR')
    preprocess(args.mat_path, args.out_dir)


if __name__ == '__main__':
    main()
