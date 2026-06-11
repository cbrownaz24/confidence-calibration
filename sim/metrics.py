"""Evaluation metrics. The headline quantity is the truthfulness MAE."""

from __future__ import annotations

import torch


@torch.no_grad()
def mae_to_p(ensemble, z_eval: torch.Tensor, p_eval: torch.Tensor) -> float:
    """Mean absolute error of the deterministic reports against the true p.

        MAE = (1 / (B*N)) sum_{b,i} | M_i(z_i) - p_i |

    This is the noise-free convergence metric the notes ask for: we *know* the
    true N-dimensional probability vector p, so distance-to-p measures exactly
    how close each agent's reporting map is to the truthful identity.
    """
    bids = ensemble.bids(z_eval)
    return float((bids - p_eval).abs().mean())


@torch.no_grad()
def per_agent_mae(ensemble, z_eval: torch.Tensor, p_eval: torch.Tensor) -> list:
    bids = ensemble.bids(z_eval)
    return (bids - p_eval).abs().mean(dim=0).cpu().tolist()
