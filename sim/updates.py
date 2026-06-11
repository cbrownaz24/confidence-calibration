"""Policy-gradient advantage estimators shared by the REINFORCE / GRPO family.

All of these turn a per-agent reward matrix ``r`` of shape [B, N] into an
*advantage* of the same shape (detached), which the algorithm then combines with
the action log-probabilities as ``loss = -(adv * logp).mean()``. The reward rule
itself (what makes a bid good) lives in the algorithm file; these helpers only
encode the *baseline / update* choice.
"""

from __future__ import annotations

import torch


def rloo_advantage(r: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline across the N agents in a round.

        A_i = r_i - mean_{j != i} r_j

    Valid as a baseline whenever the agents' rewards are exchangeable. Gives
    every agent a signal (unlike winner-only), including the "you should have bid
    higher" gradient that the auction's winner channel alone cannot supply.
    """
    n = r.shape[1]
    if n == 1:
        return (r - r.mean(dim=1, keepdim=True)).detach()
    loo = (r.sum(dim=1, keepdim=True) - r) / (n - 1)
    return (r - loo).detach()


def grpo_advantage(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative standardized advantage (GRPO).

        A_i = (r_i - mean_j r_j) / (std_j r_j + eps)

    The group here is the N agents acting on one round.
    """
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    return ((r - mean) / (std + eps)).detach()


def winner_mask(bids: torch.Tensor) -> torch.Tensor:
    """1/0 mask selecting the highest bidder per row (the auction winner)."""
    m = torch.zeros_like(bids)
    rows = torch.arange(bids.shape[0], device=bids.device)
    m[rows, bids.argmax(dim=1)] = 1.0
    return m
