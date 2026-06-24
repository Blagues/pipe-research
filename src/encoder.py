import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_enc(T: int, d: int) -> torch.Tensor:
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe  = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _CrossAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn    = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff      = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d))
        self.drop    = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_normed = self.norm_kv(kv)
        out, _    = self.attn(self.norm_q(q), kv_normed, kv_normed)
        q = q + self.drop(out)
        q = q + self.drop(self.ff(self.norm_ff(q)))
        return q


class _ProjectionHead(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DualHeadCSIEncoder(nn.Module):
    """
    Cross-attention encoder over 3 receivers × T frames × 4 antennas.
    Returns (z, p): z = raw embedding (B, d), p = L2-normalised projection (B, d).
    """
    def __init__(self, R=3, T=100, A=4, S=114, d=256, N=2, M=2,
                 mlp_dim=512, n_heads=4, ffn_dim=512,
                 mlp_dropout=0.0, self_attn_dropout=0.0,
                 cross_attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.R, self.T, self.d = R, T, d
        self.mlp = nn.Sequential(
            nn.Linear(A * S, mlp_dim), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, d), nn.LayerNorm(d),
        )
        self.register_buffer("pos_time", _sinusoidal_enc(T, d))
        self.rx_emb         = nn.Parameter(torch.randn(R, d) * 0.02)
        self.last_frame_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=ffn_dim,
                                       batch_first=True, norm_first=True, dropout=self_attn_dropout),
            num_layers=N,
        )
        self.cls_pose   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cross_attn = nn.ModuleList([
            _CrossAttnBlock(d, n_heads, ffn_dim, cross_attn_dropout) for _ in range(M)
        ])
        self.proj_pose = _ProjectionHead(d, proj_dropout)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = x.reshape(B, self.R, self.T, -1)
        x = self.mlp(x)
        x = x + self.pos_time
        x = x + self.rx_emb[None, :, None, :]
        x[:, :, -1, :] += self.last_frame_emb
        x = x.reshape(B, self.R * self.T, self.d)
        x = self.self_attn(x)
        q = self.cls_pose.expand(B, -1, -1)
        for block in self.cross_attn:
            q = block(q, x)
        z = q[:, 0]
        return z, self.proj_pose(z)
