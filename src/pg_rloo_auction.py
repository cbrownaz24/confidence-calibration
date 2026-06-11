"""Second-price auction under a symmetric RLOO update.

Realized second-price reward (winner gets y_w - second_price, losers 0), but the
advantage is leave-one-out across agents: A_i = r_i - mean_{j!=i} r_j. Every
agent updates, and losers get the "you should have bid higher" gradient that
winner-only lacks -- the symmetric update is what makes the auction usable as an
elicitation device. Source: train_auction_rloo (auction-confidence); the RLOO
corner of the DESIGN.md matrix.
"""
from sim import (Algorithm, register_algorithm, hard_auction, rloo_advantage)


@register_algorithm
class RLOOAuction(Algorithm):
    name = "pg_rloo_auction"
    family = "reinforce"
    label = "2nd-price + RLOO"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        r = hard_auction(b, y)["reward"]
        adv = rloo_advantage(r)
        return -(adv * logp).mean()
