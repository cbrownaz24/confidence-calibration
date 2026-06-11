"""ECE over training -- the overconfidence experiment.

Companion to ``convergence_mae``: instead of MAE-to-p it plots the Expected
Calibration Error against *binary outcomes* over training steps. Watching ECE
after the MAE has flattened is the small-scale reproduction of the Guo et al.
(2017) finding -- networks keep improving their point predictions while their
calibration degrades, i.e. they get overconfident with more training.

One line per algorithm, colored by family (same palette as the MAE plot).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sim import register_plot, FAMILY_COLORS


def _shade(base_hex: str, t: float):
    r, g, b = mcolors.to_rgb(base_hex)
    lo, hi = 0.55, 1.15
    f = lo + (hi - lo) * t
    return tuple(min(1.0, max(0.0, c * f)) for c in (r, g, b))


@register_plot("ece_convergence")
def plot(results: list[dict], out_dir: str, cfg=None) -> str:
    fam_order = ["score", "counterfactual", "reinforce", "grpo", "misc"]
    by_fam: dict[str, list] = {}
    for res in results:
        if not res.get("ece"):
            continue
        by_fam.setdefault(res["family"], []).append(res)
    ordered = [f for f in fam_order if f in by_fam] + \
              [f for f in by_fam if f not in fam_order]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for fam in ordered:
        members = sorted(by_fam[fam], key=lambda r: r["final_ece"])
        base = FAMILY_COLORS.get(fam, FAMILY_COLORS["misc"])
        n = len(members)
        for j, res in enumerate(members):
            t = 0.5 if n == 1 else j / (n - 1)
            ax.plot(res["iters"], res["ece"], color=_shade(base, t),
                    lw=1.9, label=res["label"])

    ax.set_xlabel("training step")
    ax.set_ylabel("ECE  (held-out, vs binary outcomes)")
    ax.set_title("Calibration over training: does confidence stay honest?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ece_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
