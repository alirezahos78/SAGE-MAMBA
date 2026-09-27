"""Common MNE processing, matching the supplied DREAMER implementation."""

from pathlib import Path
import numpy as np
import mne

Z_BAD, RATIO_BAD, MAX_ITER = 5.0, 20.0, 8
PTP_Z = 6.0


def make_info(names, fs, montage_path=None):
    info = mne.create_info(names, fs, 'eeg')
    if montage_path is not None:
        info.set_montage(mne.channels.read_custom_montage(str(montage_path)),
                         on_missing='raise', match_case=False)
        return info
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                     on_missing='ignore', match_case=False)
    if 'CB1' in names or 'CB2' in names:
        # SEED CB electrodes are not in standard_1020. These are approximate
        # coordinates; provide the recording montage for exact interpolation.
        positions = info.get_montage().get_positions()
        pos = {k.upper(): v.copy() for k, v in positions['ch_pos'].items()
               if np.isfinite(v).all()}
        for cb, occ in [('CB1', 'O1'), ('CB2', 'O2')]:
            p = pos[occ].copy()
            q = p + np.array([0.0, -0.015, -0.045])
            pos[cb] = q * np.linalg.norm(p) / np.linalg.norm(q)
        montage = mne.channels.make_dig_montage(
            ch_pos={ch: pos[ch.upper()] for ch in names},
            nasion=positions['nasion'], lpa=positions['lpa'], rpa=positions['rpa'],
            coord_frame='head')
        info.set_montage(montage)
    return info


def detect_and_interpolate(raw, *, include_flat=False, max_bad=None):
    names, ever_bad = raw.ch_names, set()
    for _ in range(MAX_ITER):
        variance = np.var(raw.get_data() * 1e6, axis=1)
        med_v = max(np.median(variance), 1e-12)
        lv = np.log10(np.maximum(variance, 1e-12))
        med = np.median(lv)
        mad = np.median(np.abs(lv - med)) * 1.4826
        z = (lv - med) / max(mad, 1e-9)
        new = [i for i in range(len(names))
               if (z[i] > Z_BAD and variance[i] / med_v > RATIO_BAD)
               or (include_flat and variance[i] <= 1e-8)]
        if not new or (max_bad is not None and len(ever_bad) >= max_bad):
            break
        new = sorted(new, key=lambda i: -variance[i])[:2]
        bad_names = [names[i] for i in new]
        raw.info['bads'] = bad_names
        raw.interpolate_bads(reset_bads=True, verbose=False)
        ever_bad.update(bad_names)
    v = np.var(raw.get_data() * 1e6, axis=1)
    return raw, sorted(ever_bad), float(v.max() / max(np.median(v), 1e-12))


def clean_signal(data_uv, info, *, include_flat=False, max_bad=None):
    data_uv = np.asarray(data_uv, dtype=np.float64)
    if data_uv.ndim != 2 or data_uv.shape[0] != len(info['ch_names']):
        raise ValueError('Expected a channels-by-samples array in microvolts.')
    if not np.isfinite(data_uv).all():
        raise ValueError('Raw signal contains NaN or infinity.')
    raw = mne.io.RawArray(data_uv * 1e-6, info.copy(), verbose=False)
    raw.notch_filter(freqs=[50.0], picks='eeg', verbose=False)
    raw.filter(l_freq=1.0, h_freq=47.0, picks='eeg', method='fir',
               fir_design='firwin', verbose=False)
    raw, bads, ratio = detect_and_interpolate(
        raw, include_flat=include_flat, max_bad=max_bad)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw.get_data() * 1e6, bads, ratio


def window_signal(data, fs, reject=True):
    win = int(fs)
    n = data.shape[1] // win
    if n <= 0:
        return np.empty((0, data.shape[0], win), np.float32), np.zeros(0, bool)
    ep = np.stack([data[:, i * win:(i + 1) * win] for i in range(n)])
    keep = np.ones(n, bool)
    if reject and n >= 10:
        ptp = np.median(np.ptp(ep, axis=2), axis=1)
        med = np.median(ptp)
        mad = np.median(np.abs(ptp - med)) * 1.4826
        if mad > 1e-9:
            keep = np.abs((ptp - med) / mad) < PTP_Z
    return ep[keep].astype(np.float32), keep


def prepare_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f'{path} is not empty; use a new output directory.')
    return path
