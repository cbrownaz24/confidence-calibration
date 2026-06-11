"""REINFORCE where losers update at reward 0 (exp2).

Same winner reward as winner-only, but every loser also takes a REINFORCE step
with reward 0. That contributes a constant-baseline shift but no informative
gradient through the action channel, so it does not actually fix the starvation
problem -- a useful contrast to the surplus / RLOO variants. Source: exp2
(alignment-auction).
"""
from sim import Algorithm, register_algorithm, hard_auction


@register_algorithm
class LosersZeroReinforce(Algorithm):
    name = "pg_losers_zero"
    family = "reinforce"
    label = "Losers-zero REINFORCE"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        r = hard_auction(b, y)["reward"]          # losers already 0 here
        adv = self.ema_advantage(r, ctx)
        return -(adv * logp).mean()               # every agent updates
