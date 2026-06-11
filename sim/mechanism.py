"""Shared mechanism primitives used by the auction-style algorithms.

These are the *reusable* pieces of the second-price machinery -- the highest
competing bid, the realized hard auction, the differentiable soft win indicator,
and the generic counterfactual loss. Individual algorithms in ``src/`` only have
to say how the competing price ``s_i`` is formed; they delegate the rest here.
"""

from __future__ import annotations

import torch


def max_of_others(bids: torch.Tensor) -> torch.Tensor:
    """For each agent i, the highest competing bid s_i = max_{j != i} b_j.

    bids: [B, N] -> [B, N]. (For the row-argmax column this is the 2nd highest;
    for every other column it is the row max.)
    """
    B, N = bids.shape
    if N == 1:
        return torch.zeros_like(bids)
    top2 = torch.topk(bids, k=2, dim=1)
    top1_val, top1_idx = top2.values[:, 0:1], top2.indices[:, 0:1]
    top2_val = top2.values[:, 1:2]
    s = top1_val.expand(B, N).clone()
    s.scatter_(1, top1_idx, top2_val)             # arg-max agent faces the 2nd bid
    return s


def hard_auction(bids: torch.Tensor, y: torch.Tensor) -> dict:
    """Run the exact second-price mechanism over a [B, N] bid matrix.

    Returns winner [B], second_price [B], and reward [B, N] where the winner gets
    ``y_winner - second_price`` and every loser gets 0.
    """
    B, N = bids.shape
    winner = bids.argmax(dim=1)
    rows = torch.arange(B, device=bids.device)
    s = max_of_others(bids)
    second_price = s[rows, winner]
    reward = torch.zeros_like(bids)
    reward[rows, winner] = y[rows, winner] - second_price
    return {"winner": winner, "second_price": second_price, "reward": reward,
            "competing": s}


def soft_win(bids: torch.Tensor, s: torch.Tensor, tau: float) -> torch.Tensor:
    """Differentiable relaxation of the win indicator 1{b_i > s_i}."""
    return torch.sigmoid((bids - s) / tau)


def counterfactual_loss(bids: torch.Tensor, s: torch.Tensor, y: torch.Tensor,
                        tau: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The differentiable counterfactual second-price loss.

        L = - mean_B  sum_i  mask_i * sigmoid((b_i - s_i)/tau) * (y_i - s_i)

    ``s`` must already be detached (opponents are held fixed for agent i). The
    optimum of each per-agent term is truthful bidding b_i = p_i. ``mask`` (1/0)
    selects which agents receive a training signal this round (used by sleep).
    """
    win = soft_win(bids, s, tau)
    term = win * (y - s)
    if mask is not None:
        num = (term * mask).sum(0)
        den = mask.sum(0).clamp(min=1.0)
        return -(num / den).sum()
    return -term.mean(0).sum()
