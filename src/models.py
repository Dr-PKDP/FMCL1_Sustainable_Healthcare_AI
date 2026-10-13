"""
models.py -- the ResNet1D architecture used in the experiment of
Section 5.2.4.

It takes a two-lead ECG window (shape: batch x 2 x window_len) and returns
logits over the five AAMI arrhythmia superclasses (N, S, V, F, Q).

GroupNorm is used in place of BatchNorm1d throughout. BatchNorm1d computes
running statistics that update on every forward pass during training and
get frozen for use at evaluation time. Under the recording-level client split
used here (one MIT-BIH recording per client), a local training round
can run on a small batch drawn from a single recording, so BatchNorm's
running mean/variance drift on that one unrepresentative client every
round, and by evaluation time those frozen statistics can be close to
noise regardless of how well the underlying convolutional weights actually
learned. GroupNorm normalises each example independently over its own
channels, with no batch-size or cross-round running-statistic dependency,
and is the standard fix for BatchNorm's known incompatibility with small
per-client batches in federated learning.
"""
import torch.nn as nn
import torch.nn.functional as F


def norm1d(ch: int) -> nn.Module:
    """GroupNorm with the largest group count in {8, 4, 2, 1} that divides
    ch evenly, falling back to a single group (equivalent to LayerNorm over
    channels) when none of 8/4/2 divide it."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ResBlock1D(nn.Module):
    def __init__(self, ch: int, kernel: int = 7):
        super().__init__()
        pad = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            norm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            norm1d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ResNet1D(nn.Module):
    """
    1D residual CNN for multi-lead ECG classification. Compact variant,
    approximately 300k parameters: a 15-wide stem convolution followed by
    6 residual blocks at 64 channels, a global average pool, and a linear
    classification head.
    """
    def __init__(self, n_leads: int = 2, n_classes: int = 5,
                 n_blocks: int = 6, base_ch: int = 64, kernel: int = 7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, 15, padding=7, bias=False),
            norm1d(base_ch), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(base_ch, kernel)
                                      for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_ch, n_classes)

    def forward(self, x):
        # x: (B, n_leads, window_len)
        z = self.stem(x)
        z = self.blocks(z)
        return self.fc(self.pool(z).squeeze(-1))
