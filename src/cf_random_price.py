"""Counterfactual auction against a uniform random price every round (eps=1).

The competing price is s_i ~ U(0,1) on every round, removing live competition
entirely. A random-price round has expected reward
E_r[(y - r) 1{b > r}] = b*y - b^2/2 = -1/2 (b - y)^2 + const, i.e. exactly the
Brier score -- so this is the auction-shaped reduction to a proper scoring rule.
It gives full-support coverage and is robust for any value distribution.
Source: PRICE_RULES["random_price"] (spauction-rafael); `sp_solo_reserve`
("Vickrey vs reserve == rand. Brier") in multi_agent_auction_calibration.
"""
import torch
from sim import Algorithm, register_algorithm, counterfactual_loss


@register_algorithm
class CounterfactualRandomPrice(Algorithm):
    name = "cf_random_price"
    family = "counterfactual"
    label = "Random price (= Brier)"

    def price(self, bids, ctx):
        return torch.rand(bids.shape, generator=ctx.gen)

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        s = self.price(b, ctx)
        return counterfactual_loss(b, s, y, ctx.tau)
