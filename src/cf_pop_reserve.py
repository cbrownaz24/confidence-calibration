"""Counterfactual auction with a population (pooled-bid) reserve.

Same as cf_reserve, but the reserve is drawn from the pooled current bids
(shuffled, so it is value-independent and decorrelated). This makes the reserve
*distribution* equal to the bid distribution, which gives full-support price
coverage matched to where agents actually bid -- the best-performing auction
refinement in the source sweep. Source: PRICE_RULES["pop_reserve"]
(spauction-rafael, flagged "best auction rule").
"""
import torch
from sim import Algorithm, register_algorithm, max_of_others, counterfactual_loss


@register_algorithm
class CounterfactualPopReserve(Algorithm):
    name = "cf_pop_reserve"
    family = "counterfactual"
    label = "Counterfactual + pop. reserve"

    def price(self, bids, ctx):
        det = bids.detach()
        base = max_of_others(det)
        flat = det.reshape(-1)
        idx = torch.randint(0, flat.numel(), (flat.numel(),), generator=ctx.gen)
        reserve = flat[idx].reshape(det.shape)
        hit = torch.rand(det.shape, generator=ctx.gen) < ctx.eps_reserve
        return torch.where(hit, reserve, base)

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        s = self.price(b, ctx)
        return counterfactual_loss(b, s, y, ctx.tau)
