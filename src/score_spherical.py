"""Proper scoring rule: spherical  (y*b + (1-y)*(1-b)) / sqrt(b^2 + (1-b)^2).

Strictly proper and bounded -- a middle ground between Brier (gentle) and log
(harsh). Source: REWARDS.md sec.4 (auction-confidence).
"""
import torch
from sim import Algorithm, register_algorithm


@register_algorithm
class Spherical(Algorithm):
    name = "spherical"
    family = "score"
    label = "Spherical (proper)"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        num = y * b + (1 - y) * (1 - b)
        den = torch.sqrt(b ** 2 + (1 - b) ** 2)
        return -(num / den).mean(0).sum()
