"""Run the sparse backbones and a separately trained phi=1 Mamba comparison."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['seed', 'dreamer'])
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--device', choices=['auto', 'cuda'], default='auto')
    parser.add_argument('--attention-batches', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    common = [sys.executable, '-m', f'sage_mamba.cli.train_{args.dataset}',
              '--data-dir', args.data_dir, '--out-dir', args.out_dir,
              '--epochs', str(args.epochs), '--seeds', *map(str, args.seeds),
              '--device', args.device, '--attention-batches', str(args.attention_batches),
              '--num-workers', str(args.num_workers)]
    subprocess.run(common + ['--phi', '0.3', '--backbones', 'mamba', 'lstm', 'gru'], check=True)
    subprocess.run(common + ['--phi', '1.0', '--backbones', 'mamba'], check=True)
    if not args.no_plots:
        out = Path(args.out_dir)
        seed = 42 if 42 in args.seeds else args.seeds[0]
        targets = ['inter_session'] if args.dataset == 'seed' else ['valence', 'arousal']
        for target in targets:
            artifact = f'inter_subject_mamba_{target}_seed{seed}.npz'
            subprocess.run([
                sys.executable, '-m', 'sage_mamba.cli.plot_artifacts',
                '--artifact', str(out / 'phi_0p3' / 'artifacts' / artifact),
                '--dense', str(out / 'phi_1p0' / 'artifacts' / artifact),
                '--out-dir', str(out / 'figures' / target)], check=True)
        subprocess.run([sys.executable, '-m', 'sage_mamba.cli.draw_architecture',
                        '--out-dir', str(out / 'figures')], check=True)


if __name__ == '__main__':
    main()
