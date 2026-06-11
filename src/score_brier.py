"""Proper scoring rule: Brier  -(b - y)^2.

Strictly proper, bounded, gentle. The expected reward under true p is uniquely
maximized at b = p, so the differentiable pathwise gradient drives M_i(p) -> p.
Source: REWARDS.md sec.2 (auction-confidence); `proper_brier` in both notebooks.
"""
from sim import Algorithm, register_algorithm


@register_algorithm
class Brier(Algorithm):
    name = "brier"
    family = "score"
    label = "Brier (proper)"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        return ((b - y) ** 2).mean(0).sum()
