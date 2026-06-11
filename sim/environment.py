"""The simulation environment (Section 3 of the Confidence Notes).

For each simulated prompt we draw a vector of private correctness probabilities

    (p_1, ..., p_N) ~ F,        p_i in [0, 1],

hand each agent its own ``p_i`` (optionally as a noisy observation ``z_i``), and
sample exogenous correctness

    y_i ~ Bernoulli(p_i).

The environment is deliberately the *fixed* part of the repository: it exposes a
small, stable API (``sample_p`` / ``observe`` / ``sample_y``) that every
algorithm in ``src/`` consumes. Algorithms never touch the sampling internals.
"""

from __future__ import annotations

import torch

from .config import EnvConfig


def _resolve_beta_params(cfg: EnvConfig, n: int):
    if cfg.alphas is not None and cfg.betas is not None:
        return (torch.tensor(cfg.alphas, dtype=torch.float32),
                torch.tensor(cfg.betas, dtype=torch.float32))
    # spread agents across the skill spectrum: some weak, some strong.
    return torch.linspace(1.0, 5.0, n), torch.linspace(5.0, 1.0, n)


class Environment:
    """Generates (p, z, y) tuples, each of shape [B, N]."""

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.n = cfg.n_agents
        self._gen = torch.Generator().manual_seed(seed)
        if cfg.kind == "beta":
            self.alphas, self.betas = _resolve_beta_params(cfg, self.n)
        elif cfg.kind == "correlated":
            if cfg.a is not None:
                self.a = torch.tensor(cfg.a, dtype=torch.float32)
            else:
                self.a = torch.randn(self.n, generator=self._gen) * cfg.a_std

    # ---------------------------------------------------------------- values
    def sample_p(self, batch: int) -> torch.Tensor:
        g, k = self._gen, self.cfg.kind
        if k == "uniform":
            p = torch.rand(batch, self.n, generator=g)
        elif k == "beta":
            a = self.alphas.expand(batch, self.n)
            b = self.betas.expand(batch, self.n)
            ga = torch._standard_gamma(a, generator=g)
            gb = torch._standard_gamma(b, generator=g)
            p = ga / (ga + gb)
        elif k == "correlated":
            q = torch.rand(batch, 1, generator=g)
            zq = (q - 0.5) * 4.0
            eps = torch.randn(batch, self.n, generator=g) * self.cfg.eps_std
            logits = self.a.expand(batch, self.n) + self.cfg.gamma * zq + eps
            p = torch.sigmoid(logits)
        else:
            raise ValueError(f"unknown env kind: {k!r}")
        return p.clamp(1e-4, 1 - 1e-4)

    # --------------------------------------------------------------- signals
    def observe(self, p: torch.Tensor) -> torch.Tensor:
        """The agent observation z given latent p (z == p unless signal_std>0)."""
        if self.cfg.signal_std <= 0.0:
            return p
        noise = torch.randn(p.shape, generator=self._gen) * self.cfg.signal_std
        return (p + noise).clamp(1e-4, 1 - 1e-4)

    # -------------------------------------------------------------- outcomes
    def sample_y(self, p: torch.Tensor) -> torch.Tensor:
        u = torch.rand(p.shape, generator=self._gen)
        return (u < p).float()
