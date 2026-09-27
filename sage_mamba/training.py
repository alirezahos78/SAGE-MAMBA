"""Training, table export, and numeric graph/attention artifacts."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sage_mamba.channels import channel_names
from sage_mamba.data import PROTOCOLS, SEED_CONDITIONS, load_dreamer, load_seed, split_indices
from sage_mamba.experiment_io import atomic_json, atomic_npz, index_hash, phi_directory, sha256_file
from sage_mamba.model import ModelConfig, SAGENet, expected_parameter_count, parameter_count

torch.set_num_threads(1)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        j = self.indices[index]
        return torch.from_numpy(np.asarray(self.X[j], dtype=np.float32)), int(self.y[j])


def precision_context(device, dataset, backbone):
    if device.type != 'cuda' or (dataset == 'seed' and backbone != 'mamba'):
        return nullcontext()
    dtype = (torch.bfloat16 if backbone == 'mamba' and torch.cuda.is_bf16_supported()
             else torch.float16)
    return torch.autocast('cuda', dtype=dtype)


def evaluate(model, loader, device, dataset, backbone, subject, test_indices):
    model.eval()
    predicted, actual = [], []
    with torch.no_grad(), precision_context(device, dataset, backbone):
        for xb, yb in loader:
            predicted.append(model(xb.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            actual.append(yb.numpy())
    predicted, actual = np.concatenate(predicted), np.concatenate(actual)
    test_subject = subject[test_indices]
    per_subject = {str(int(s)): float(accuracy_score(actual[test_subject == s],
                                                   predicted[test_subject == s]))
                   for s in np.unique(test_subject)}
    return dict(pooled_accuracy=float(accuracy_score(actual, predicted)),
                mean_subject_accuracy=float(np.mean(list(per_subject.values()))),
                macro_f1=float(f1_score(actual, predicted, average='macro', zero_division=0)),
                per_subject=per_subject,
                confusion=confusion_matrix(actual, predicted,
                                           labels=list(range(model.config.n_classes))).tolist())


def average_attention(model, loader, device, dataset, backbone, max_batches=0):
    total = np.zeros(model.config.n_channels, np.float64)
    count = 0
    model.eval()
    with torch.no_grad(), precision_context(device, dataset, backbone):
        for batch_index, (xb, _) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            weights = model.features(xb.to(device, non_blocking=True))[1]
            weights = weights.squeeze(-1).float().cpu().numpy()
            total += weights.sum(axis=0, dtype=np.float64)
            count += len(weights)
    if count == 0:
        raise ValueError('No test windows available for attention export.')
    return (total / count).astype(np.float32), count


def train_one(data, dataset, target, selected, protocol, backbone, seed, args, out):
    """One independently initialized model for a complete protocol/condition cell."""
    device = torch.device(args.device)
    if backbone == 'mamba' and device.type != 'cuda':
        raise RuntimeError('Mamba training requires CUDA; select lstm/gru for CPU checks.')
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_indices, test_indices = split_indices(data.subject, selected, protocol, seed)
    y = data.labels['emotion' if dataset == 'seed' else target]
    train_loader = torch.utils.data.DataLoader(
        WindowDataset(data.X, y, train_indices), batch_size=64, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
        drop_last=len(train_indices) > 128)
    test_loader = torch.utils.data.DataLoader(
        WindowDataset(data.X, y, test_indices), batch_size=128, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    cfg = ModelConfig.for_dataset(dataset, backbone, args.phi)
    model = SAGENet(cfg).to(device)
    n_params = parameter_count(model)
    if n_params != expected_parameter_count(cfg):
        raise RuntimeError(f'Unexpected parameter count: {n_params}; check dependency versions.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp_fp16 = (device.type == 'cuda' and
                (dataset == 'dreamer' or backbone == 'mamba') and
                not (backbone == 'mamba' and torch.cuda.is_bf16_supported()))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_fp16)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=max(1, args.epochs * len(train_loader)))
    metric = 'mean_subject_accuracy' if protocol == 'intra_subject' else 'pooled_accuracy'
    tag = f'{protocol}_{backbone}_{target}_seed{seed}'
    best, best_state, best_epoch = -1.0, None, None
    history, start = [], time.time()
    print(f'{tag} | phi={args.phi} | parameters={n_params:,} | '
          f'train/test={len(train_indices)}/{len(test_indices)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device, dataset, backbone):
                loss = criterion(model(xb), yb) + 5e-4 * model.l1_penalty()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{tag}: nonfinite loss at epoch {epoch}.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach()) * len(xb)
            count += len(xb)
        scores = evaluate(model, test_loader, device, dataset, backbone,
                          data.subject, test_indices)
        score = scores[metric]
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=total_loss / count,
                            test_accuracy=score, pooled_accuracy=scores['pooled_accuracy'],
                            mean_subject_accuracy=scores['mean_subject_accuracy']))
        print(f'  epoch {epoch:2d}/{args.epochs} | loss={total_loss/count:.4f} | '
              f'test={score*100:.2f}% | best={best*100:.2f}%', flush=True)
    model.load_state_dict(best_state)
    scores = evaluate(model, test_loader, device, dataset, backbone, data.subject, test_indices)
    metadata = dict(dataset=dataset, backbone=backbone, seed=seed, protocol=protocol,
                    phi=float(args.phi), d_model=cfg.d_model, n_parameters=n_params,
                    selected_epoch=best_epoch, n_test_windows=len(test_indices),
                    train_split_sha256=index_hash(train_indices),
                    test_split_sha256=index_hash(test_indices))
    metadata['condition' if dataset == 'seed' else 'target'] = target
    if args.save_checkpoints:
        ckpt = out / 'checkpoints' / f'{tag}.pt'
        torch.save(dict(state_dict=best_state, config=asdict(cfg), metadata=metadata),
                   ckpt.with_suffix('.tmp'))
        os.replace(ckpt.with_suffix('.tmp'), ckpt)
    if args.save_artifacts:
        attention, n_attention = average_attention(
            model, test_loader, device, dataset, backbone, args.attention_batches)
        artifact_metadata = dict(metadata, n_attention_windows=n_attention,
                                 attention_aggregation='mean_over_test_windows')
        atomic_npz(out / 'artifacts' / f'{tag}.npz',
                   attention=attention, channel_names=np.asarray(channel_names(dataset)),
                   attention_indices=test_indices[:n_attention],
                   **artifact_metadata,
                   **{f'adjacency_block_{i}': graph for i, graph in enumerate(model.adjacencies())})
    atomic_npz(out / 'splits' / f'{tag}.npz', train_indices=train_indices, test_indices=test_indices)
    result = dict(metadata, **scores, accuracy=scores[metric], n_train_windows=len(train_indices),
                  selected_by=f'test_{metric}', epochs=args.epochs,
                  elapsed_minutes=(time.time() - start) / 60, history=history)
    atomic_json(out / 'runs' / f'{tag}.json', result)
    del model, best_state, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return tag, result


def write_tables(cells, dataset, targets, seeds, backbones, out):
    order = [b for b in ['gru', 'lstm', 'mamba'] if b in backbones]
    labels = {'gru': 'SAGE-GRU', 'lstm': 'SAGE-LSTM', 'mamba': 'SAGE-Mamba'}
    header = (['Model', 'Inter session', 'Session 1', 'Session 2', 'Session 3']
              if dataset == 'seed' else ['Model', 'Valence (%)', 'Arousal (%)'])
    all_targets = SEED_CONDITIONS if dataset == 'seed' else ['valence', 'arousal']
    for protocol in PROTOCOLS:
        path = out / f'{dataset}_{protocol}_table.csv'
        with path.open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for backbone in order:
                row = [labels[backbone]]
                for target in all_targets:
                    values = [100 * cells[f'{protocol}_{backbone}_{target}_seed{seed}']['accuracy']
                              for seed in seeds
                              if f'{protocol}_{backbone}_{target}_seed{seed}' in cells]
                    row.append(f'{np.mean(values):.2f} ± {np.std(values, ddof=0):.2f}'
                               if len(values) == len(seeds)
                               else f'pending ({len(values)}/{len(seeds)})')
                writer.writerow(row)


def parser_for(dataset):
    parser = argparse.ArgumentParser(description=f'Train SAGE models on {dataset.upper()}.')
    parser.add_argument('--data-dir', default='checkpoints' if dataset == 'seed' else 'DREAMER_processed')
    parser.add_argument('--out-dir', default=f'results/{dataset}')
    parser.add_argument('--backbones', nargs='+', choices=['mamba', 'lstm', 'gru'],
                        default=['mamba', 'lstm', 'gru'])
    parser.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=PROTOCOLS)
    if dataset == 'seed':
        parser.add_argument('--conditions', nargs='+', choices=SEED_CONDITIONS, default=SEED_CONDITIONS)
    else:
        parser.add_argument('--targets', nargs='+', choices=['valence', 'arousal'],
                            default=['valence', 'arousal'])
    parser.add_argument('--phi', type=float, default=0.30)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--attention-batches', type=int, default=0,
                        help='0 exports attention over the full test split; 8 uses at most 1024 windows.')
    parser.add_argument('--save-artifacts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--check-data', action='store_true', help='Validate data and show dimensions without training.')
    return parser


def main(dataset):
    args = parser_for(dataset).parse_args()
    if args.epochs < 1 or not 0 < args.phi <= 1 or args.attention_batches < 0 or args.num_workers < 0:
        raise ValueError('Invalid epochs, phi, attention-batches, or num-workers.')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError('Seeds must be distinct integers in [0, 2**32).')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this environment.')
    data = (load_seed if dataset == 'seed' else load_dreamer)(args.data_dir)
    print(f'{dataset.upper()}: X={data.X.shape}, dtype={data.X.dtype}, '
          f'subjects={len(np.unique(data.subject))}', flush=True)
    if args.check_data:
        for target, y in data.labels.items():
            print(f'{target}: class counts {np.bincount(y).tolist()}')
        return
    if 'mamba' in args.backbones and args.device != 'cuda':
        raise RuntimeError('Use a CUDA environment for Mamba, or select --backbones lstm gru.')
    out = Path(args.out_dir) / phi_directory(args.phi)
    for child in ['artifacts', 'checkpoints', 'runs', 'splits']:
        (out / child).mkdir(parents=True, exist_ok=True)
    print('Computing input checksums for safe resume...', flush=True)
    root = Path(__file__).parent
    manifest = dict(dataset=dataset, phi=args.phi, epochs=args.epochs,
                    attention_batches=args.attention_batches,
                    save_artifacts=args.save_artifacts, save_checkpoints=args.save_checkpoints,
                    device=args.device, torch_version=str(torch.__version__),
                    files=[dict(name=p.name, sha256=sha256_file(p)) for p in data.files],
                    source={name: sha256_file(root / name) for name in
                            ['model.py', 'training.py', 'data.py', 'experiment_io.py']})
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Output directory belongs to a different configuration/data/source. '
                           'Use a new --out-dir to avoid mixing results.')
    atomic_json(manifest_path, manifest)
    cache_path = out / 'cells.json'
    cells = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    targets = args.conditions if dataset == 'seed' else args.targets
    if dataset == 'seed':
        jobs = [(b, p, t, s) for b in args.backbones for p in args.protocols
                for t in targets for s in args.seeds]
    else:
        jobs = [(b, p, t, s) for p in args.protocols for s in args.seeds
                for t in targets for b in args.backbones]
    for backbone, protocol, target, seed in jobs:
        tag = f'{protocol}_{backbone}_{target}_seed{seed}'
        if tag in cells:
            expected = [out / 'runs' / f'{tag}.json', out / 'splits' / f'{tag}.npz']
            if args.save_artifacts:
                expected.append(out / 'artifacts' / f'{tag}.npz')
            if args.save_checkpoints:
                expected.append(out / 'checkpoints' / f'{tag}.pt')
            if not all(path.exists() for path in expected):
                raise FileNotFoundError(f'Incomplete cached outputs for {tag}; use a new output directory.')
            print(f'Cached: {tag}', flush=True)
        else:
            selected = np.arange(len(data.X))
            if dataset == 'seed' and target != 'inter_session':
                selected = np.flatnonzero(data.session == int(target[-1]))
            key, result = train_one(data, dataset, target, selected, protocol,
                                    backbone, seed, args, out)
            cells[key] = result
            atomic_json(cache_path, cells)
        write_tables(cells, dataset, targets, args.seeds, args.backbones, out)
    atomic_json(out / 'table_summary.json', dict(seeds=args.seeds, std_ddof=0,
                backbones=args.backbones, protocols=args.protocols, targets=targets,
                selection='best test epoch; no validation split'))
    print(f'Results: {out.resolve()}', flush=True)
