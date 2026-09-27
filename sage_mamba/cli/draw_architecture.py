"""Draw an editable vector overview directly from the implemented operations."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set(xlim=(0, 10.6), ylim=(0, 5.6))
    ax.axis('off')

    def box(x, y, w, h, text, color, size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=.08',
                                    edgecolor='#394958', facecolor=color, linewidth=1.05))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                     linewidth=1.1, color='#394958'))

    box(.15, 4.1, 1.8, .8, 'EEG window\n$X$: $C \\times T$', '#edf1f5')
    box(2.4, 4.1, 2.1, .8, 'Channel-wise Conv1D\npatch size = stride = 4', '#e8f1fb', 9)
    box(5.0, 4.1, 2.2, .8, 'Graph-temporal block\nrepeated $L=4$ times', '#e8f5ef', 10)
    box(7.75, 4.1, 2.5, .8, 'Temporal mean\n$C \\times D$ descriptors', '#fcf0d8')
    for a, b in [((1.95, 4.5), (2.4, 4.5)), ((4.5, 4.5), (5, 4.5)), ((7.2, 4.5), (7.75, 4.5))]:
        arrow(a, b)
    ax.text(5.3, 5.28, 'SAGE-Mamba', ha='center', fontsize=17, weight='bold')
    # The inset spells out both residual paths without changing their ordering.
    ax.add_patch(FancyBboxPatch((.18, .35), 6.75, 3.22, boxstyle='round,pad=.07',
                               facecolor='#f9fbfa', edgecolor='#728779', linestyle='--'))
    ax.text(3.55, 3.28, 'Inside each graph-temporal block', ha='center', fontsize=11)
    box(.48, 2.12, 2.5, .67, 'Learned signed affinity\nsymmetric tanh + top-$\\phi$ mask', '#e3f1e8', 9)
    box(3.67, 2.12, 2.83, .67, 'Self loops + row normalization\ntwo graph linear layers + GELU', '#e3f1e8', 9)
    arrow((2.98, 2.46), (3.67, 2.46))
    ax.text(5.1, 1.89, 'Add graph input, then LayerNorm', ha='center', fontsize=8.5)
    box(3.67, .82, 2.83, .69, 'Forward + reverse Mamba\nconcatenate, then project to $D$', '#e5ecfb', 9)
    arrow((5.1, 1.76), (5.1, 1.51))
    box(.48, .82, 2.5, .69, 'LayerNorm temporal output\nadd graph-branch output', '#e5ecfb', 9)
    arrow((3.67, 1.16), (2.98, 1.16))
    box(7.75, 2.57, 2.5, .82, 'Shared channel MLP\n$D \\to 32 \\to 1$ with tanh\nsoftmax over $C$ channels', '#fcf0d8', 9)
    box(7.75, 1.38, 2.5, .68, 'Attention-weighted sum\n$D$-dimensional vector', '#fcf0d8', 9)
    box(7.75, .2, 2.5, .68, 'Classifier\nemotion logits', '#f4e8ed', 10)
    arrow((9, 4.1), (9, 3.39))
    arrow((9, 2.57), (9, 2.06))
    arrow((9, 1.38), (9, .88))
    ax.text(.2, .03, '$C$: channels; $T$: input samples; $D$: embedding width; '
             '$L$: number of blocks; $\\phi$: retained-affinity threshold fraction.', fontsize=8)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'architecture.{suffix}', dpi=300, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
    print(f'Architecture saved to {out.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures')
    draw(parser.parse_args().out_dir)
