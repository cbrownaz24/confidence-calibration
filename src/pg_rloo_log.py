"""RLOO with a logarithmic proper-scoring reward (no auction).

Each agent's reward is the log score of its own (bid, outcome); a leave-one-out
baseline across agents reduces variance (valid because pointwise rewards do not
depend on others' bids). Being a strictly proper rule, the optimum is truthful
b_i = p_i regardless of the value distribution or competition -- it sidesteps the
auction entirely while staying in the policy-gradient regime. Source:
train_log_rloo (auction-confidence, flagged "mine").
"""
import torch
from sim import Algorithm, register_algorithm, rloo_advantage


@register_algorithm
class RLOOLog(Algorithm):
    name = "pg_rloo_log"
    family = "reinforce"
    label = "Log + RLOO"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        bc = b.clamp(1e-4, 1 - 1e-4)
        r = y * torch.log(bc) + (1 - y) * torch.log(1 - bc)   # log score
        adv = rloo_advantage(r)
        return -(adv * logp).mean()
