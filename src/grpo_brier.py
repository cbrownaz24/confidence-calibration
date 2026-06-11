"""Brier proper-scoring reward under a GRPO update.

A strictly proper per-agent reward (-(b-y)^2) combined with the group-relative
standardized GRPO advantage across the N agents. Pairs the calibration-friendly
proper reward with the policy-gradient update used in the GRPO notebook; a useful
point of comparison against the dense pathwise Brier and the auction-under-GRPO
variant. Source: `proper_brier` under grpo_train (auction_grpo_calibration).
"""
from sim import (Algorithm, register_algorithm, grpo_advantage)


@register_algorithm
class GRPOBrier(Algorithm):
    name = "grpo_brier"
    family = "grpo"
    label = "Brier + GRPO"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        r = -((b - y) ** 2)                        # Brier reward, per agent
        adv = grpo_advantage(r)
        return -(adv * logp).mean()
