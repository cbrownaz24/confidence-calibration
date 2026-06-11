"""Supervised identity  (b - p)^2  (diagnostic ceiling).

Uses the *latent* p directly, which a real elicitation setup never sees, so this
is not a deployable mechanism -- it is the best-case reference curve that every
truthful mechanism is trying to match. Source: SCORE_RULES["supervised"]
(spauction-rafael); the oracle baseline in alignment-auction.
"""
from sim import Algorithm, register_algorithm


@register_algorithm
class Supervised(Algorithm):
    name = "supervised"
    family = "score"
    label = "Supervised (oracle ceiling)"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)
        return ((b - p) ** 2).mean(0).sum()
