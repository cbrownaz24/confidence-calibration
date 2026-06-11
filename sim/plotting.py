"""Plot registry. Each plot type is a drop-in file in ``plots/``.

A plot module defines a function decorated with :func:`register_plot` that takes
the list of per-algorithm result dicts (as returned by ``Algorithm.train``) plus
the output directory, renders a figure, and returns the path it wrote. Adding a
new kind of plot is exactly: drop a new file in ``plots/`` with one decorated
function. ``plot_experiments.py`` discovers and runs whichever ones are enabled
in the config.
"""

from __future__ import annotations

from typing import Callable

PLOT_REGISTRY: dict[str, Callable] = {}


def register_plot(name: str):
    def deco(fn: Callable) -> Callable:
        if name in PLOT_REGISTRY and PLOT_REGISTRY[name] is not fn:
            raise ValueError(f"duplicate plot name {name!r}")
        PLOT_REGISTRY[name] = fn
        return fn
    return deco


# Stable color per algorithm family, with per-member variation handled by the
# plot itself. Kept here so all plots share one palette.
FAMILY_COLORS = {
    "score":          "#4C72B0",   # proper scoring rules (differentiable)
    "counterfactual": "#55A868",   # differentiable soft second-price auctions
    "reinforce":      "#C44E52",   # REINFORCE / policy-gradient auctions
    "grpo":           "#8172B2",   # GRPO group-relative updates
    "misc":           "#937860",
}
