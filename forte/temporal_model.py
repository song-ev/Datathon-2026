"""Small temporal network over frozen visual features, including within-video changes."""
import numpy as np
import torch
from torch import nn


def transform(features, pca):
    x = features.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    x = pca.transform(x).astype(np.float32)
    relative = x - np.median(x, axis=0, keepdims=True)
    before = np.stack([x[max(0, i-3):i+1].mean(0) for i in range(len(x))])
    after = np.stack([x[i:min(len(x), i+4)].mean(0) for i in range(len(x))])
    return np.concatenate([x, relative, after-before], axis=1)


class TemporalDetector(nn.Module):
    def __init__(self, dimensions=96):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(dimensions, 32, 3, padding=1), nn.GELU(),
                                 nn.Dropout(.2), nn.Conv1d(32, 16, 3, padding=1),
                                 nn.GELU(), nn.Conv1d(16, 1, 1))

    def forward(self, x):
        return self.net(x.T[None])[0, 0]


def clip_logit(logits):
    # Require sustained support across up to three sampled frames.
    return logits.topk(min(3, len(logits))).values.mean()
