"""
Soft InfoNCE contrastive loss and MoCap-based similarity matrices.
"""
import torch


def soft_similarity_matrix(coords: torch.Tensor, sigma: float) -> torch.Tensor:
    """Pairwise similarity on joint positions."""
    D = (coords.unsqueeze(1) - coords.unsqueeze(0)).pow(2).sum(-1).sqrt()
    S = torch.exp(-D / sigma)
    S.fill_diagonal_(0.0)
    return S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)


def soft_similarity_matrix_vel(vel: torch.Tensor, sigma_v: float) -> torch.Tensor:
    """Pairwise similarity on joint velocities."""
    D_v = (vel.unsqueeze(1) - vel.unsqueeze(0)).pow(2).sum(-1).sqrt()
    S_v = torch.exp(-D_v / sigma_v)
    S_v.fill_diagonal_(0.0)
    return S_v / S_v.sum(dim=1, keepdim=True).clamp(min=1e-8)


def soft_similarity_matrix_dual(coords: torch.Tensor, vel: torch.Tensor,
                                 sigma_p: float, sigma_v: float) -> torch.Tensor:
    """Product of pose and velocity similarities"""
    D_p = (coords.unsqueeze(1) - coords.unsqueeze(0)).pow(2).sum(-1).sqrt()
    S_p = torch.exp(-D_p / sigma_p)
    D_v = (vel.unsqueeze(1) - vel.unsqueeze(0)).pow(2).sum(-1).sqrt()
    S_v = torch.exp(-D_v / sigma_v)
    S   = S_p * S_v
    S.fill_diagonal_(0.0)
    return S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)


def soft_info_nce(p: torch.Tensor, S: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    logits = (p @ p.T) / tau
    logits.fill_diagonal_(float("-inf"))
    log_p  = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(torch.where(S > 0, S * log_p, torch.zeros_like(log_p))).sum(dim=1).mean()
