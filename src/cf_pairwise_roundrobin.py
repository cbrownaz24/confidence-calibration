"""Round-robin pairwise second-price auctions (dense reward for every agent).

The starvation failure of the peer-priced second-price auction is that a lagging
agent sits where opponents never set its price, so it sees no gradient. This rule
densifies the signal: instead of one N-way auction, each agent i plays a separate
two-player second-price auction against *every* other agent j, and its loss is
averaged over all N-1 of those matchups. In a 2-player second price, i wins iff
b_i > b_j and pays b_j, so the counterfactual reward against opponent j is

    sigmoid((b_i - b_j)/tau) * (y_i - b_j),

and every agent gets a gradient from N-1 contests every round -- no one is
starved. Optimum of each pairwise term is still truthful b_i = p_i.

Starting point for the overnight search (see RESEARCH_BRIEF.md). Natural variants
to try: weight matchups, add a reserve to each pairwise contest, or pool the
pairwise prices.
"""
import torch
from sim import Algorithm, register_algorithm


@register_algorithm
class PairwiseRoundRobin(Algorithm):
    name = "cf_pairwise_roundrobin"
    family = "counterfactual"
    label = "Pairwise round-robin"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)                       # [B, N], column i carries grad
        bj = b.detach().unsqueeze(1)               # [B, 1, N] opponents (fixed)
        bi = b.unsqueeze(2)                         # [B, N, 1] me (grad)
        win = torch.sigmoid((bi - bj) / ctx.tau)   # [B, N, N]
        payoff = y.unsqueeze(2) - bj               # [B, N, N]
        term = win * payoff                         # reward of i vs each j
        # average over opponents j != i (zero the diagonal, divide by N-1)
        N = b.shape[1]
        eye = torch.eye(N, device=b.device).bool().unsqueeze(0)
        term = term.masked_fill(eye, 0.0).sum(dim=2) / max(N - 1, 1)
        return -term.mean(0).sum()
