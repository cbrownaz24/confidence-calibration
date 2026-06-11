"""Configuration for the confidence-calibration simulation.

Everything that defines *the simulation* (environment, agents, training budget)
lives here as typed dataclasses and is loaded from a single YAML file in
``config/``. The same ``ExperimentConfig`` is handed to every algorithm so that
each one runs against an identical simulation -- the only thing that varies
between curves on the final plot is the algorithm in ``src/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml


@dataclass
class EnvConfig:
    """How the latent correctness probabilities ``p_i`` are generated.

    kind:
      - "uniform"    : p_i ~ U(0, 1), i.i.d. across agents (independent values).
      - "beta"       : p_i ~ Beta(alpha_i, beta_i), heterogeneous skill per agent.
      - "correlated" : shared difficulty q, p_i = sigmoid(a_i + gamma*z(q) + eps_i)
                       -- the affiliated-values regime (winner's-curse territory).
      - "hidden_features" : the *probability is never revealed*. A shared question
                       feature x in R^d is drawn each round; agent i's hidden
                       correctness is c_i(x) = sigmoid(gain*f_i(x) + skill_i) where
                       f_i is a frozen random teacher net, and only the binary
                       outcome y_i ~ Bernoulli(c_i(x)) is observed. The agent must
                       *infer* its confidence from binary feedback alone -- it
                       never sees c_i. (See FeatureEnvironment.)
    """

    kind: str = "uniform"
    n_agents: int = 5
    # beta params (per-agent lists, length N); auto-generated if None.
    alphas: Optional[list] = None
    betas: Optional[list] = None
    # correlated params
    a: Optional[list] = None
    gamma: float = 2.0
    eps_std: float = 0.5
    a_std: float = 1.0
    # observation noise: z = clip(p + N(0, signal_std)).  0 => z == p (default).
    signal_std: float = 0.0

    # ---- hidden_features parameters (used only by kind == "hidden_features") ----
    feature_dim: int = 8          # dimension d of the question feature x in R^d
    teacher_kind: str = "mlp"     # "mlp" (rich, frozen random net) | "linear"
    teacher_hidden: tuple = (16, 16)   # hidden widths of each agent's teacher f_i
    teacher_gain: float = 1.5     # logit scale; bigger => more confident teacher
    teacher_seed: int = 12345     # seeds the frozen teachers (independent of agents)
    skill_spread: float = 1.5     # per-agent skill bias spread (strong..weak agents)
    label_noise: float = 0.0      # eta: outcome flipped w.p. eta (Bayes target shifts)


@dataclass
class AgentConfig:
    """The per-agent MLP. Every agent is an independent network of this shape."""

    hidden: tuple = (32, 32)
    activation: str = "tanh"      # "relu" or "tanh"
    output: str = "sigmoid"       # "sigmoid" (robust) or "clamp"
    init_bid: float = 0.5         # interior initial report for every agent


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256
    lr: float = 5e-3
    eval_every: int = 50
    eval_batch: int = 20000

    # soft-threshold temperature for the differentiable counterfactual auctions,
    # annealed tau0 -> tau1 over training.
    tau0: float = 0.3
    tau1: float = 0.03

    # exploration std (logit space) for stochastic / REINFORCE / GRPO bids,
    # annealed sigma0 -> sigma_min over the first 80% of training.
    sigma0: float = 0.7
    sigma_min: float = 0.15

    anneal: bool = True

    # REINFORCE / GRPO knobs
    baseline_decay: float = 0.95  # EMA baseline decay
    adv_clip: float = 3.0         # advantage clip (+/-)

    # auction-refinement knobs (used by the counterfactual price rules)
    eps_reserve: float = 0.2      # random-reserve probability
    eps_sleep: float = 0.5        # sleep probability for the sleep mechanism


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # selection of algorithms / plots, and output location
    algorithms: dict = field(default_factory=lambda: {"enabled": "all", "disabled": []})
    plots: dict = field(default_factory=lambda: {"enabled": "all", "disabled": []})
    output: dict = field(default_factory=lambda: {"dir": "outputs"})


# --------------------------------------------------------------------------- #
# nested dataclass fields, resolved by name (annotations are strings under
# ``from __future__ import annotations``, so we can't rely on f.type directly).
_NESTED = {"env": EnvConfig, "agent": AgentConfig, "train": TrainConfig}


def _coerce(dc_type, data: dict[str, Any]):
    """Recursively build a dataclass ``dc_type`` from a (possibly nested) dict,
    ignoring unknown keys and keeping defaults for missing ones."""
    if data is None:
        return dc_type()
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        if f.name in _NESTED and isinstance(val, dict):
            kwargs[f.name] = _coerce(_NESTED[f.name], val)
        elif f.name in ("hidden", "teacher_hidden") and isinstance(val, list):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def load_config(path: str) -> ExperimentConfig:
    """Load the experiment config from a YAML file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _coerce(ExperimentConfig, raw)
