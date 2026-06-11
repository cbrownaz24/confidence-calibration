"""The algorithm blueprint that every file in ``src/`` follows.

To add a new confidence-calibration algorithm you write one small file in
``src/`` that subclasses :class:`Algorithm`, sets ``name`` / ``family``, and
implements a single method, :meth:`Algorithm.compute_loss`. Everything else --
the optimizer per agent, the annealed schedules, the evaluation loop that records
the MAE-to-p convergence curve, and the registration that makes the algorithm
show up in the plotting script -- is handled here and is identical across
algorithms. That uniformity is what keeps the simulation constant: the only thing
that differs between two curves is the loss returned by ``compute_loss``.

Minimal example::

    from sim import Algorithm, register_algorithm

    @register_algorithm
    class MyRule(Algorithm):
        name = "my_rule"
        family = "score"

        def compute_loss(self, batch, ctx):
            p, z, y = batch
            b = self.ens.bids(z)
            return ((b - y) ** 2).mean(0).sum()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .agents import AgentEnsemble
from .config import ExperimentConfig
from .environment import Environment
from .metrics import mae_to_p, per_agent_mae

# --------------------------------------------------------------------------- #
# Registry: every @register_algorithm subclass lands here, keyed by ``name``.
REGISTRY: dict[str, type["Algorithm"]] = {}


def register_algorithm(cls: type["Algorithm"]) -> type["Algorithm"]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must set a class-level `name`")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate algorithm name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class StepCtx:
    """Per-step context handed to ``compute_loss`` (schedules + rng)."""
    step: int
    total_steps: int
    tau: float            # soft-threshold temperature (counterfactual auctions)
    sigma: float          # exploration std (REINFORCE / GRPO bids)
    eps_reserve: float
    eps_sleep: float
    gen: torch.Generator


class Algorithm:
    """Base class. Subclasses override :meth:`compute_loss` (and class attrs)."""

    name: str = "base"            # unique id, also the legend label key
    family: str = "misc"          # grouping for plot styling
    label: Optional[str] = None   # optional pretty legend label

    def __init__(self, ensemble: AgentEnsemble, env: Environment,
                 cfg: ExperimentConfig, gen: torch.Generator):
        self.ens = ensemble
        self.env = env
        self.cfg = cfg
        self.gen = gen
        # one optimizer per agent: each agent learns independently.
        self.opts = [torch.optim.Adam(a.parameters(), lr=cfg.train.lr)
                     for a in ensemble.agents]
        # EMA reward baseline, available to REINFORCE-style subclasses.
        self.baseline = torch.zeros(env.n)

    # ----------------------------------------------------------- to override
    def compute_loss(self, batch, ctx: StepCtx) -> torch.Tensor:
        """Return a scalar loss for one round. ``batch = (p, z, y)`` each [B, N]."""
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    def ema_advantage(self, r: torch.Tensor, ctx: StepCtx) -> torch.Tensor:
        """EMA-baselined, clipped advantage for plain REINFORCE updates."""
        clip = self.cfg.train.adv_clip
        adv = (r - self.baseline).detach().clamp(-clip, clip)
        with torch.no_grad():
            d = self.cfg.train.baseline_decay
            self.baseline = d * self.baseline + (1 - d) * r.mean(dim=0)
        return adv

    def _schedules(self, step: int):
        t = self.cfg.train
        if not t.anneal or t.steps <= 1:
            return t.tau0, t.sigma0
        frac = step / max(t.steps - 1, 1)
        tau = t.tau0 + (t.tau1 - t.tau0) * frac
        # exploration decays over the first 80% of training, then holds.
        sfrac = min(1.0, step / max(t.steps * 0.8, 1))
        sigma = t.sigma0 + (t.sigma_min - t.sigma0) * sfrac
        return tau, sigma

    # ----------------------------------------------------------------- loop
    def train(self, z_eval: torch.Tensor, p_eval: torch.Tensor,
              verbose: bool = False) -> dict:
        t = self.cfg.train
        iters, curve = [], []
        for step in range(t.steps + 1):
            if step % t.eval_every == 0 or step == t.steps:
                iters.append(step)
                curve.append(mae_to_p(self.ens, z_eval, p_eval))
            if step == t.steps:
                break
            tau, sigma = self._schedules(step)
            ctx = StepCtx(step, t.steps, tau, sigma,
                          t.eps_reserve, t.eps_sleep, self.gen)
            p = self.env.sample_p(t.batch_size)
            z = self.env.observe(p)
            y = self.env.sample_y(p)
            loss = self.compute_loss((p, z, y), ctx)
            for o in self.opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in self.opts:
                o.step()
        result = {
            "name": self.name,
            "label": self.label or self.name,
            "family": self.family,
            "iters": iters,
            "mae": curve,
            "final_mae": curve[-1],
            "per_agent_mae": per_agent_mae(self.ens, z_eval, p_eval),
        }
        if verbose:
            print(f"  {self.name:24s} final MAE={curve[-1]:.4f}", flush=True)
        return result


def build_simulation(cfg: ExperimentConfig):
    """Construct a fresh, identically-seeded environment + agent ensemble.

    Re-seeding here (per algorithm) guarantees every algorithm sees the *same*
    stream of (p, z, y) and starts from the *same* agent initialisation, so the
    simulation is held constant across all curves.
    """
    torch.manual_seed(cfg.seed)
    env = Environment(cfg.env, seed=cfg.seed)
    ens = AgentEnsemble(cfg.env.n_agents, cfg.agent)
    gen = torch.Generator().manual_seed(1000 + cfg.seed)
    return env, ens, gen


def run_algorithm(alg_cls: type[Algorithm], cfg: ExperimentConfig,
                  verbose: bool = False) -> dict:
    """Run one algorithm end-to-end against a constant simulation."""
    env, ens, gen = build_simulation(cfg)
    # fixed held-out evaluation set (identical across algorithms).
    p_eval = env.sample_p(cfg.train.eval_batch)
    z_eval = env.observe(p_eval)
    alg = alg_cls(ens, env, cfg, gen)
    return alg.train(z_eval, p_eval, verbose=verbose)
