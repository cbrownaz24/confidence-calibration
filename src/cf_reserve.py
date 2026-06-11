"""Counterfactual second-price auction with a random reserve.

With probability eps_reserve the competing price is replaced by an independent
reserve r ~ U(0,1); otherwise it is the usual second price max_{j!=i} b_j. The
reserve injects an independent, full-support competitor, restoring the clean
proper-scoring optimum (truthful bidding) even when live competition is
degenerate (one dominant agent / skewed value distributions). Payment never
depends on the agent's own bid, so truthful bidding stays dominant.
Source: PRICE_RULES["reserve"] (spauction-rafael); eps-reserve (auction-confidence).
"""
import torch
from sim import Algorithm, register_algorithm, max_of_others, counterfactual_loss


@register_algorithm
class CounterfactualReserve(Algorithm):
    name = "cf_reserve"
    family = "counterfactual"
    label = "Counterfactual + reserve"

    def price(self, bids, ctx):
        base = max_of_others(bids.detach())
        hit = torch.rand(bids.shape, generator=ctx.gen) < ctx.eps_reserve
        reserve = torch.rand(bids.shape, generator=ctx.gen)
        return torch.where(hit, reserve, base)

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        s = self.price(b, ctx)
        return counterfactual_loss(b, s, y, ctx.tau)
