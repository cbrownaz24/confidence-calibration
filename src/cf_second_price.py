"""Counterfactual second-price auction (bare).

Every agent gets a *differentiable counterfactual* gradient: the reward it would
get for bidding b_i against the held-fixed competing price s_i = max_{j!=i} b_j,
with the hard win indicator relaxed to a sigmoid. Unlike winner-only REINFORCE,
losers also learn. Optimum is truthful b_i = p_i under independent values.
Source: PRICE_RULES["bare"] (spauction-rafael); train_auction (auction-confidence).
"""
from sim import Algorithm, register_algorithm, max_of_others, counterfactual_loss


@register_algorithm
class CounterfactualSecondPrice(Algorithm):
    name = "cf_second_price"
    family = "counterfactual"
    label = "Counterfactual 2nd-price"

    def price(self, bids, ctx):
        return max_of_others(bids.detach())

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        s = self.price(b, ctx)
        return counterfactual_loss(b, s, y, ctx.tau)
