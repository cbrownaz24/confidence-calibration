"""The fixed simulation package.

Public surface used by algorithms in ``src/`` and by the plotting script. This
API is intended to stay stable across the lifespan of the repository.
"""

from .config import (
    ExperimentConfig, EnvConfig, AgentConfig, TrainConfig, load_config,
)
from .environment import Environment
from .feature_environment import FeatureEnvironment
from .agents import AgentEnsemble, MLPAgent
from .mechanism import (
    max_of_others, hard_auction, soft_win, counterfactual_loss,
)
from .updates import rloo_advantage, grpo_advantage, winner_mask
from .metrics import mae_to_p, per_agent_mae, ece, reliability_curve
from .algorithm import (
    Algorithm, StepCtx, register_algorithm, REGISTRY,
    build_simulation, run_algorithm,
)
from .plotting import register_plot, PLOT_REGISTRY, FAMILY_COLORS

__all__ = [
    "ExperimentConfig", "EnvConfig", "AgentConfig", "TrainConfig", "load_config",
    "Environment", "FeatureEnvironment", "AgentEnsemble", "MLPAgent",
    "max_of_others", "hard_auction", "soft_win", "counterfactual_loss",
    "rloo_advantage", "grpo_advantage", "winner_mask",
    "mae_to_p", "per_agent_mae", "ece", "reliability_curve",
    "Algorithm", "StepCtx", "register_algorithm", "REGISTRY",
    "build_simulation", "run_algorithm",
    "register_plot", "PLOT_REGISTRY", "FAMILY_COLORS",
]
