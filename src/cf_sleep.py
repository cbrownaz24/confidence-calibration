"""Counterfactual auction with stochastic participation (eps-sleep).

Each round, each agent sleeps with probability eps_sleep: it is removed from the
competition and its loss is masked out (no update). An awake agent's competing
price is the second price among the *awake* opponents only. Random thinning of
the field diversifies the prices each agent faces, helping perennial winners and
losers alike see a useful gradient. Source: PRICE_RULES["sleep"]
(spauction-rafael); the "sleep" method in identity_convergence (alignment-auction).
"""
import torch
from sim import Algorithm, register_algorithm, counterfactual_loss

NEG = -1e9


@register_algorithm
class CounterfactualSleep(Algorithm):
    name = "cf_sleep"
    family = "counterfactual"
    label = "Counterfactual + sleep"

    def price_and_mask(self, bids, ctx):
        det = bids.detach()
        B, N = det.shape
        awake = torch.rand(det.shape, generator=ctx.gen) >= ctx.eps_sleep
        masked = torch.where(awake, det, torch.full_like(det, NEG))
        s = torch.empty_like(det)
        for i in range(N):
            others = torch.cat([masked[:, :i], masked[:, i + 1:]], dim=1)
            s[:, i] = others.max(dim=1).values.clamp(min=0.0)
        return s, awake.float()

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        s, mask = self.price_and_mask(b, ctx)
        return counterfactual_loss(b, s, y, ctx.tau, mask=mask)
