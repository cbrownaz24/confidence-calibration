"""The hidden-probability environment: confidence must be *inferred*, not read.

This is the same N-agent confidence-calibration game as :mod:`sim.environment`,
with one decisive change: **the probability is never revealed to the agent.**

In the original environment each agent observes its own latent ``p_i`` (the
observation is literally ``z = p``), so "confidence" is a trivial relabeling of an
input the agent already holds. Here, instead, every round draws a single shared
*question feature* ``x in R^d``. Agent ``i`` carries a frozen, randomly-initialised
*teacher* network ``f_i`` (its private, unknown skill), and its hidden correctness
probability is

    c_i(x) = sigmoid(gain * f_i(x) + skill_i).

The agent sees only ``x`` and the realised binary outcome ``y_i ~ Bernoulli(c_i(x))``.
It never observes ``c_i``. The only way to report a meaningful confidence is to
*infer* ``c_i`` from the structure of ``x`` and the binary feedback -- which is the
actual epistemic situation a model is in when it must say how likely it is to be
right. Because the teacher is smooth in ``x``, structure is shared across questions
and a single binary sample per question already carries signal (no need to resample
the same question many times).

The probability vector ``c(x)`` (optionally tilted by label noise) is still known
to *us*, so the headline MAE-to-``p`` metric and the calibration metrics (ECE,
reliability) can be evaluated exactly on held-out questions -- the calibration is
defined on novel inputs, which is the phenomenon of interest.

API parity with :class:`sim.environment.Environment`: both expose ``sample_batch``
and ``eval_set``, so the training loop and every algorithm in ``src/`` consume the
two environments interchangeably.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import EnvConfig


class _Teacher(nn.Module):
    """A frozen random map f_i: R^d -> R (an agent's unknown private skill).

    Never trained. Initialised from a dedicated generator so the teachers are
    fixed regardless of agent initialisation order or seed, giving every
    algorithm the identical hidden difficulty landscape.
    """

    def __init__(self, d: int, hidden: tuple, kind: str, gen: torch.Generator):
        super().__init__()
        if kind == "linear":
            dims = [d]
        else:
            dims = [d, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)
        # deterministic init from the teacher generator, then freeze.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * std)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=gen) * 0.1)
        for prm in self.parameters():
            prm.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, d] -> [B, 1]
        return self.head(self.body(x))


class FeatureEnvironment:
    """Generates (p, x, y): hidden per-agent probability, shared feature, outcome.

    Shapes per round of batch ``B``:
        x : [B, d]      one shared question feature (all agents see the same x)
        p : [B, N]      hidden correctness c_i(x) (the Bayes target; never shown)
        y : [B, N]      observed binary outcomes y_i ~ Bernoulli(p_i)
    """

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.n = cfg.n_agents
        self.d = cfg.feature_dim
        self.gain = cfg.teacher_gain
        self.eta = cfg.label_noise
        # stream generator: drives x-sampling and outcome draws (per-algorithm
        # seeded, so all algorithms see the identical question/outcome stream).
        self._gen = torch.Generator().manual_seed(seed)
        # separate stream for the held-out eval set, so evaluation never perturbs
        # the training RNG stream.
        self._eval_gen = torch.Generator().manual_seed(seed + 7777)
        # teacher generator: fixed across algorithms so the hidden difficulty
        # landscape itself never changes between runs.
        tgen = torch.Generator().manual_seed(cfg.teacher_seed)
        self.teachers = nn.ModuleList([
            _Teacher(self.d, cfg.teacher_hidden, cfg.teacher_kind, tgen)
            for _ in range(self.n)
        ])
        # per-agent skill bias: spread agents from strong (+) to weak (-), the
        # hidden-feature analogue of the beta-spread skill heterogeneity.
        s = cfg.skill_spread
        self.skill = torch.linspace(s, -s, self.n) if self.n > 1 else torch.zeros(1)

    # ---------------------------------------------------------------- features
    def _sample_x(self, batch: int, gen: torch.Generator) -> torch.Tensor:
        return torch.randn(batch, self.d, generator=gen)

    @torch.no_grad()
    def _c_of_x(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden correctness c_i(x) for every agent: [B, d] -> [B, N]."""
        logits = torch.cat([t(x) for t in self.teachers], dim=1)  # [B, N]
        logits = self.gain * logits + self.skill
        return torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)

    def _p_target(self, c: torch.Tensor) -> torch.Tensor:
        """Bayes-optimal confidence target given label noise eta.

        With outcomes flipped w.p. eta, the achievable correctness rate is
        ``(1 - eta) c + eta (1 - c)`` -- which is what a calibrated agent should
        report and what y is actually drawn from.
        """
        if self.eta <= 0.0:
            return c
        return (1.0 - self.eta) * c + self.eta * (1.0 - c)

    def _draw(self, batch: int, gen: torch.Generator):
        x = self._sample_x(batch, gen)
        p = self._p_target(self._c_of_x(x))
        u = torch.rand(p.shape, generator=gen)
        y = (u < p).float()
        return p, x, y

    # ------------------------------------------------------------- public API
    def sample_batch(self, batch: int):
        """One training round: returns (p, x, y) with p hidden from algorithms."""
        return self._draw(batch, self._gen)

    def eval_set(self, batch: int):
        """A held-out evaluation set (drawn first, so it is identical across runs).

        Returns (x_eval, p_eval, y_eval): the observation handed to ``ens.bids``,
        the true hidden probability target, and a fixed outcome draw for ECE.
        Drawn entirely from the dedicated eval stream so it leaves the training
        stream untouched (and is identical across algorithms).
        """
        p, x, y = self._draw(batch, self._eval_gen)
        return x, p, y
