#!/usr/bin/env python3
"""Run every enabled algorithm against the constant simulation, then plot.

Workflow (the whole point of the repo):
    1. think of a new algorithm for N agents to become confidence-calibrated,
    2. drop a new file in src/ (subclass Algorithm, implement compute_loss),
    3. run `python plot_experiments.py`,
    4. get the convergence plot with your new algorithm in the legend.

This script auto-discovers everything in src/ (algorithms) and plots/ (figures),
filters by the enabled/disabled lists in the config, runs each algorithm against
the *same* environment / agent / training config, and renders each enabled plot.

Usage:
    python plot_experiments.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import sys
import time

# make the repo importable regardless of where the script is launched from
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim import (load_config, run_algorithm, REGISTRY, PLOT_REGISTRY)


def _discover(package: str) -> None:
    """Import every submodule of ``package`` so its @register_* decorators run."""
    pkg = importlib.import_module(package)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{mod.name}")


def _resolve(enabled, disabled, available: list[str], what: str) -> list[str]:
    disabled = set(disabled or [])
    if enabled in (None, "all", ["all"]):
        chosen = [n for n in available if n not in disabled]
    else:
        chosen = []
        for n in enabled:
            if n not in available:
                raise SystemExit(f"unknown {what} {n!r}; available: {available}")
            if n not in disabled:
                chosen.append(n)
    if not chosen:
        raise SystemExit(f"no {what}s selected")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join("config", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)

    # populate the registries
    _discover("src")
    _discover("plots")

    alg_names = _resolve(cfg.algorithms.get("enabled"),
                         cfg.algorithms.get("disabled"),
                         sorted(REGISTRY), "algorithm")
    plot_names = _resolve(cfg.plots.get("enabled"),
                          cfg.plots.get("disabled"),
                          sorted(PLOT_REGISTRY), "plot")

    out_dir = cfg.output.get("dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"simulation: env={cfg.env.kind} N={cfg.env.n_agents} "
          f"steps={cfg.train.steps} seed={cfg.seed}")
    print(f"running {len(alg_names)} algorithms: {', '.join(alg_names)}\n")

    results = []
    t0 = time.time()
    for name in alg_names:
        results.append(run_algorithm(REGISTRY[name], cfg, verbose=True))
    print(f"\ntrained {len(results)} algorithms in {time.time() - t0:.1f}s")

    # persist the raw curves so plots can be re-rendered without retraining
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nrendering plots:")
    for name in plot_names:
        path = PLOT_REGISTRY[name](results, out_dir, cfg)
        print(f"  {name:20s} -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
