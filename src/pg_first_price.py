"""First-price auction under winner-only REINFORCE.

The winner pays its own (highest) bid: reward = y_w - max_i b_i. Because the bid
sets *what you pay*, truthful bidding is no longer dominant -- the incentive is to
shade bids below value -- so this is an improper-incentive contrast to the
second-price rule. Source: `first_price` in both notebooks.
"""
import torch
from sim import Algorithm, register_algorithm, winner_mask


@register_algorithm
class FirstPriceReinforce(Algorithm):
    name = "pg_first_price"
    family = "reinforce"
    label = "First price"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        B = b.shape[0]
        rows = torch.arange(B)
        winner = b.argmax(dim=1)
        first_price = b[rows, winner]
        r = torch.zeros_like(b)
        r[rows, winner] = y[rows, winner] - first_price   # winner pays own bid
        adv = self.ema_advantage(r, ctx)
        mask = winner_mask(b)
        return -(adv * logp * mask).sum(1).mean()
