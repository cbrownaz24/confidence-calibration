"""The agents: N independent MLPs, one per agent.

Each agent maps a scalar observation ``z in [0,1]`` (its own private value, or a
noisy signal of it) to a report / bid ``b in [0,1]``. Parameters are never shared
across agents, so gradients never mix -- agent i is updated only to improve its
own objective. The ensemble offers two ways to produce bids:

  * ``bids(z)``            -- deterministic mean report (used by the differentiable
                             score / counterfactual algorithms and by evaluation).
  * ``stochastic_bids(z)`` -- a sampled report plus its Gaussian log-probability
                             in logit space (used by the REINFORCE / GRPO
                             algorithms, which need a score-function gradient).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import AgentConfig


class MLPAgent(nn.Module):
    """M_i: observation -> report b in [0,1].

    ``input_dim`` is 1 in the classic environment (the agent observes its own
    scalar value) and ``d`` in the hidden-features environment (the agent observes
    the shared question feature x in R^d and must infer its confidence from it).
    """

    def __init__(self, cfg: AgentConfig, input_dim: int = 1):
        super().__init__()
        act = nn.ReLU if cfg.activation == "relu" else nn.Tanh
        dims = [input_dim, *cfg.hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)
        self.output = cfg.output
        # Start every agent's report in the interior near ``init_bid`` so the
        # clamp head is not trapped at a zero-gradient boundary.
        nn.init.zeros_(self.head.weight)
        if self.output == "sigmoid":
            b = cfg.init_bid
            nn.init.constant_(self.head.bias, math.log(b / (1 - b)))
        else:
            nn.init.constant_(self.head.bias, cfg.init_bid)

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(z))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.logit(z)
        if self.output == "sigmoid":
            return torch.sigmoid(raw)
        return raw.clamp(0.0, 1.0)


class AgentEnsemble(nn.Module):
    """Holds N independent MLP agents and produces a [B, N] bid matrix.

    Two feeding modes, set by ``input_dim``:
      * ``input_dim == 1`` (classic): agent ``i`` is fed its own column
        ``z[:, i]`` -- each agent observes its private scalar value.
      * ``input_dim > 1`` (hidden features): every agent is fed the *same*
        shared question feature ``x`` of shape ``[B, input_dim]``. Agents differ
        only in their learned map, so confidence must be inferred from x.
    """

    def __init__(self, n_agents: int, cfg: AgentConfig, input_dim: int = 1):
        super().__init__()
        self.n = n_agents
        self.cfg = cfg
        self.input_dim = input_dim
        self.shared_input = input_dim > 1
        self.agents = nn.ModuleList(
            [MLPAgent(cfg, input_dim) for _ in range(n_agents)])

    def _feed(self, z: torch.Tensor, i: int) -> torch.Tensor:
        """The slice of the observation handed to agent ``i``."""
        return z if self.shared_input else z[:, i:i + 1]

    def bids(self, z: torch.Tensor) -> torch.Tensor:
        """Observation -> [B, N] deterministic mean reports."""
        cols = [self.agents[i](self._feed(z, i)) for i in range(self.n)]
        return torch.cat(cols, dim=1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        cols = [self.agents[i].logit(self._feed(z, i)) for i in range(self.n)]
        return torch.cat(cols, dim=1)

    def stochastic_bids(self, z: torch.Tensor, sigma: float):
        """Sample a report in logit space; return (bids, logprob), both [B, N].

        The sample is detached (it is the fixed action); the gradient flows
        through the policy mean via ``logprob``. The clamp boundary correction is
        ignored (standard REINFORCE practice).
        """
        mu = self.logits(z)                       # [B, N], carries grad (shape via _feed)
        z_sample = (mu + torch.randn_like(mu) * sigma).detach()
        bids = torch.sigmoid(z_sample)
        logprob = (-0.5 * ((z_sample - mu) / sigma) ** 2
                   - math.log(sigma) - 0.5 * math.log(2 * math.pi))
        return bids, logprob
