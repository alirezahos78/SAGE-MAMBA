"""SAGE networks: the supplied graph/attention computation with three backbones."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_channels: int
    n_classes: int
    d_model: int
    backbone: str = 'mamba'
    phi: float = 0.30
    patch_size: int = 4
    n_blocks: int = 4
    rnn_hidden: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    min_degree: int = 2
    dropout: float = 0.5
    head_activation: str = 'relu'

    @classmethod
    def for_dataset(cls, dataset, backbone='mamba', phi=0.30):
        if dataset == 'seed':
            return cls(62, 3, 128, backbone=backbone, phi=phi)
        if dataset == 'dreamer':
            return cls(14, 2, 96, backbone=backbone, phi=phi,
                       head_activation='gelu')
        raise ValueError(f'Unknown dataset: {dataset}')


class BiMamba(nn.Module):
    def __init__(self, D, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError('Install mamba_ssm and causal_conv1d for the CUDA '
                              'environment. No alternative backbone is substituted.') from exc
        self.fwd = Mamba(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out = nn.Linear(D * 2, D)

    def forward(self, x):
        return self.out(torch.cat([self.fwd(x), self.bwd(x.flip(1)).flip(1)], -1))


class BiLSTM(nn.Module):
    def __init__(self, D, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(D, hidden, 1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, D)

    def forward(self, x):
        s, _ = self.lstm(x)
        return self.proj(s)


class BiGRU(nn.Module):
    def __init__(self, D, hidden=128):
        super().__init__()
        self.gru = nn.GRU(D, hidden, 1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, D)

    def forward(self, x):
        s, _ = self.gru(x)
        return self.proj(s)


class GraphBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.keep_frac, self.min_deg = cfg.phi, cfg.min_degree
        self.register_buffer('I', torch.eye(cfg.n_channels))
        self.A_learn = nn.Parameter(torch.zeros(cfg.n_channels, cfg.n_channels))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        D = cfg.d_model
        self.gcn1, self.gcn2 = nn.Linear(D, D), nn.Linear(D, D)
        self.norm_g = nn.LayerNorm(D)
        if cfg.backbone == 'mamba':
            self.seq = BiMamba(D, cfg.d_state, cfg.d_conv, cfg.expand)
        elif cfg.backbone == 'lstm':
            self.seq = BiLSTM(D, cfg.rnn_hidden)
        elif cfg.backbone == 'gru':
            self.seq = BiGRU(D, cfg.rnn_hidden)
        else:
            raise ValueError(f'Unknown backbone: {cfg.backbone}')
        self.norm_t = nn.LayerNorm(D)

    def symmetric(self):
        return torch.tanh((self.A_learn + self.A_learn.T) / 2)

    def sparse_adj(self):
        affinity = self.symmetric()
        if self.keep_frac >= 1.0:
            return affinity
        # This is the original global threshold, including diagonal entries.
        # Ties and the row floor can increase the realized density above phi.
        with torch.no_grad():
            mag = affinity.abs()
            k = max(1, int(round(self.keep_frac * mag.numel())))
            mask = mag >= mag.flatten().topk(k).values[-1]
            if self.min_deg > 0:
                mask = mask | (mag >= mag.topk(self.min_deg, dim=1).values[:, -1:])
            mask = (mask | mask.T).to(affinity.dtype)
        # Hard forward mask, straight-through gradient for pruned entries.
        return affinity * mask + (affinity - affinity.detach()) * (1.0 - mask)

    def adjacency(self):
        A = self.I + torch.sigmoid(self.alpha) * self.sparse_adj()
        return A / (A.abs().sum(1, keepdim=True) + 1e-6)

    def forward(self, h):
        B, Tp, C, D = h.shape
        A = self.adjacency()
        g = h.reshape(B * Tp, C, D)
        g = F.gelu(self.gcn1(A @ g))
        g = self.gcn2(A @ g)
        h = self.norm_g(h + g.reshape(B, Tp, C, D))
        s = h.permute(0, 2, 1, 3).reshape(B * C, Tp, D)
        s = self.norm_t(self.seq(s))
        s = s.reshape(B, C, Tp, D).permute(0, 2, 1, 3)
        return h + s


class SAGENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if not 0 < cfg.phi <= 1 or cfg.min_degree > cfg.n_channels:
            raise ValueError('Require 0 < phi <= 1 and min_degree <= n_channels.')
        self.config, self.D = cfg, cfg.d_model
        self.patch = nn.Conv1d(1, self.D, cfg.patch_size, stride=cfg.patch_size)
        self.blocks = nn.ModuleList([GraphBlock(cfg) for _ in range(cfg.n_blocks)])
        self.ch_attn = nn.Sequential(nn.Linear(self.D, 32), nn.Tanh(), nn.Linear(32, 1))
        activation = nn.ReLU() if cfg.head_activation == 'relu' else nn.GELU()
        self.head = nn.Sequential(
            nn.LayerNorm(self.D), nn.Dropout(cfg.dropout), nn.Linear(self.D, 64),
            activation, nn.Dropout(cfg.dropout), nn.Linear(64, cfg.n_classes))

    def features(self, x):
        B, C, T = x.shape
        if C != self.config.n_channels or T % self.config.patch_size:
            raise ValueError('Input must be (batch, channels, samples), with samples divisible by patch_size.')
        h = self.patch(x.reshape(B * C, 1, T))
        Tp = h.shape[-1]
        h = h.reshape(B, C, self.D, Tp).permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h)
        descriptors = h.mean(1)
        weights = torch.softmax(self.ch_attn(descriptors), dim=1)
        return (descriptors * weights).sum(1), weights

    def forward(self, x):
        return self.head(self.features(x)[0])

    def l1_penalty(self):
        total = 0.0
        for block in self.blocks:
            A = block.symmetric()
            total = total + (A - torch.diag(torch.diag(A))).abs().mean()
        return total / len(self.blocks)

    @torch.no_grad()
    def adjacencies(self):
        """Masked symmetric affinities, before self loops and normalization."""
        return [block.sparse_adj().float().cpu().numpy() for block in self.blocks]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_parameter_count(cfg):
    """Analytic count, including Mamba v1 default dt_rank=ceil(D/16)."""
    D, C, H = cfg.d_model, cfg.n_channels, cfg.rnn_hidden
    if cfg.backbone == 'mamba':
        inner, rank = cfg.expand * D, math.ceil(D / 16)
        branch = (3 * D * inner + inner * (cfg.d_conv + 1)
                  + inner * (rank + 2 * cfg.d_state) + inner * (rank + 1)
                  + inner * cfg.d_state + inner)
        temporal = 2 * branch + 2 * D * D + D
    else:
        gates = 4 if cfg.backbone == 'lstm' else 3
        temporal = 2 * gates * H * (D + H + 2) + 2 * H * D + D
    block = C * C + 1 + 2 * (D * D + D) + 4 * D + temporal
    patch = D * (cfg.patch_size + 1)
    attention = D * 32 + 32 + 32 + 1
    head = 2 * D + D * 64 + 64 + 64 * cfg.n_classes + cfg.n_classes
    return patch + cfg.n_blocks * block + attention + head
