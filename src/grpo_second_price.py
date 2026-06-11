"""Second-price auction under a GRPO update.

Realized second-price reward, with a group-relative standardized advantage over
the N agents in a round: A_i = (r_i - mean_j r_j) / (std_j r_j + eps). This is
the GRPO learner from auction_grpo_calibration applied to the Vickrey reward;
the sparse single-winner signal makes it noisier than the dense proper rules.
Source: `second_price` under grpo_train (auction_grpo_calibration); the grpo
corner of the DESIGN.md matrix.
"""
from sim import (Algorithm, register_algorithm, hard_auction, grpo_advantage)


@register_algorithm
class GRPOSecondPrice(Algorithm):
    name = "grpo_second_price"
    family = "grpo"
    label = "2nd-price + GRPO"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        r = hard_auction(b, y)["reward"]
        adv = grpo_advantage(r)
        return -(adv * logp).mean()
