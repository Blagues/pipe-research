"""
ActivityTransformer: sequence classifier on top of frozen encoder embeddings.
"""
import torch
import torch.nn as nn

from src.encoder import _sinusoidal_enc


class _SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = _sinusoidal_enc(max_len, d_model)       # same formula as the encoder
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ActivityTransformer(nn.Module):
    def __init__(self, d_model: int, n_classes: int, n_heads=4, n_layers=1, dropout=0.3, ffn_mult=2, max_seq_len=2000):
        super().__init__()
        self.pos_enc = _SinCosPositionalEncoding(d_model, max_len=max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        x   = self.pos_enc(x)
        out = self.transformer(x, src_key_padding_mask=mask)
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            out   = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            out = out.mean(dim=1)
        return self.head(out)
