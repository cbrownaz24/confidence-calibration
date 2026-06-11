"""Proper scoring rule: logarithmic  y*log(b) + (1-y)*log(1-b).

Strictly proper but unbounded -- punishes confident wrong reports hardest. This
is the rule from "Rewarding Doubt" (Stangel et al. 2025). Source: REWARDS.md
sec.3; `proper_log` in the notebooks.
"""
import torch
from sim import Algorithm, register_algorithm


@register_algorithm
class LogScore(Algorithm):
    name = "log"
    family = "score"
    label = "Log (proper)"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z).clamp(1e-5, 1 - 1e-5)
        return -(y * torch.log(b) + (1 - y) * torch.log(1 - b)).mean(0).sum()
