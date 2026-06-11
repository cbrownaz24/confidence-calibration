#!/usr/bin/env python3
"""Multi-seed comparison harness for algorithm search.

Runs a chosen set of algorithms over several seeds at a fast budget and reports
the mean +/- std of the final MAE-to-p, sorted best-first. This is the tool the
overnight search uses to judge a candidate *reliably*: the policy-gradient curves
in particular are noisy, so single-seed final MAE is not trustworthy -- always
compare on the seed-averaged number.

Usage:
    python search_eval.py --config config/search.yaml --seeds 3 \
        --algos supervised brier log cf_pop_reserve cf_pairwise_roundrobin

If --algos is omitted, every registered algorithm is evaluated.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import pkgutil
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim import load_config, run_algorithm, REGISTRY


def _discover(package: str) -> None:
    pkg = importlib.import_module(package)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{package}.{mod.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join("config", "search.yaml"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", nargs="*", default=None,
                    help="algorithm names; default = all registered")
    ap.add_argument("--out", default=os.path.join("search", "last_eval.json"))
    args = ap.parse_args()

    _discover("src")
    cfg = load_config(args.config)
    algos = args.algos or sorted(REGISTRY)
    for name in algos:
        if name not in REGISTRY:
            raise SystemExit(f"unknown algorithm {name!r}; have {sorted(REGISTRY)}")

    print(f"# {args.seeds} seeds x {cfg.train.steps} steps, env={cfg.env.kind} "
          f"N={cfg.env.n_agents}\n")
    rows = []
    t0 = time.time()
    for name in algos:
        finals = []
        for s in range(args.seeds):
            c = dataclasses.replace(cfg, seed=s)
            finals.append(run_algorithm(REGISTRY[name], c)["final_mae"])
        m = statistics.mean(finals)
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        rows.append({"algo": name, "mae_mean": m, "mae_std": sd, "finals": finals})
        print(f"  {name:26s} MAE {m:.4f} +/- {sd:.4f}", flush=True)

    rows.sort(key=lambda r: r["mae_mean"])
    print(f"\nranking (best first), {time.time() - t0:.0f}s:")
    for i, r in enumerate(rows, 1):
        print(f"  {i:2d}. {r['algo']:26s} {r['mae_mean']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
