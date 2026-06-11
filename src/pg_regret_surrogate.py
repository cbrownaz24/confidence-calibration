"""Regret-surrogate reward shaping under REINFORCE (exp4).

Constructs winner and loser rewards so the jump r_win - r_lose equals
(p_i - competing_price) in both branches, giving correctly-signed gradients
everywhere -- including the half of valuation space where winner-only updates
have no upward signal. Truthful bidding is the unique fixed point. Ported to the
per-agent setting (each agent has its own outcome y_i). Source: exp4 /
regret_surrogate (alignment-auction); the principled fix in algorithm.tex.

  competing price  s_i = max_{j!=i} b_j
  winner i:  (p_i > s_i) ? y_i - p_i : y_i - s_i
  loser  i:  min(s_i - p_i, 0)
"""
import torch
from sim import (Algorithm, register_algorithm, max_of_others)


@register_algorithm
class RegretSurrogate(Algorithm):
    name = "pg_regret_surrogate"
    family = "reinforce"
    label = "Regret surrogate (exp4)"

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b, logp = self.ens.stochastic_bids(z, ctx.sigma)
        s = max_of_others(b.detach())
        B = b.shape[0]
        rows = torch.arange(B)
        winner = b.argmax(dim=1)
        is_winner = torch.zeros_like(b, dtype=torch.bool)
        is_winner[rows, winner] = True

        winner_truthful = y - p                 # used where p_i > s_i
        winner_priced = y - s                   # used otherwise
        winner_r = torch.where(p > s, winner_truthful, winner_priced)
        loser_r = torch.minimum(s - p, torch.zeros_like(s))
        r = torch.where(is_winner, winner_r, loser_r)

        adv = self.ema_advantage(r, ctx)
        return -(adv * logp).mean()
