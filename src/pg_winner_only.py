"""Winner-only REINFORCE over the realized second-price auction (the failure).

Only the highest bidder updates, on its realized payoff y_w - second_price;
losers get no gradient. With no upward signal for honest low bids, reports
collapse toward the floor -- the canonical failure mode this whole study is meant
to fix. Source: exp1 (alignment-auction); `naive`/`second_price` winner-take-all
in multi_agent_auction_calibration; train_auction_stock (auction-confidence).
"""
from sim import (Algorithm, register_algorithm, hard_auction, winner_mask)


@register_algorithm
class WinnerOnlyReinforce(Algorithm):
    name = "pg_winner_only"
    family = "reinforce"
    label = "Winner-only REINFORCE"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        r = hard_auction(b, y)["reward"]          # winner: y - 2nd price, else 0
        adv = self.ema_advantage(r, ctx)
        mask = winner_mask(b)                      # only the winner is updated
        return -(adv * logp * mask).sum(1).mean()
