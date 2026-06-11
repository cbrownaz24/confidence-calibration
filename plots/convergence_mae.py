"""Headline plot: convergence of MAE-to-p over training steps, all algorithms.

Single axes, one line per algorithm (labeled in the legend), log-scaled y so the
spread between fast and slow / non-converging methods is visible. Lines are
colored by algorithm family (proper scores, counterfactual auctions, REINFORCE,
GRPO) with a per-member shade so related algorithms read as a group.

This is just one plot module. To add another kind of figure, drop a new file in
``plots/`` with a ``@register_plot("...")`` function of the same signature.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sim import register_plot, FAMILY_COLORS


def _shade(base_hex: str, t: float):
    """Return a shade of ``base_hex``; t in [0,1] maps light->dark."""
    r, g, b = mcolors.to_rgb(base_hex)
    lo, hi = 0.55, 1.15                     # lighten ... slightly darken
    f = lo + (hi - lo) * t
    return tuple(min(1.0, max(0.0, c * f)) for c in (r, g, b))


@register_plot("convergence_mae")
def plot(results: list[dict], out_dir: str, cfg=None) -> str:
    # group by family, preserve a stable family order
    fam_order = ["score", "counterfactual", "reinforce", "grpo", "misc"]
    by_fam: dict[str, list] = {}
    for res in results:
        by_fam.setdefault(res["family"], []).append(res)
    ordered = [f for f in fam_order if f in by_fam] + \
              [f for f in by_fam if f not in fam_order]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for fam in ordered:
        members = sorted(by_fam[fam], key=lambda r: r["final_mae"])
        base = FAMILY_COLORS.get(fam, FAMILY_COLORS["misc"])
        n = len(members)
        for j, res in enumerate(members):
            t = 0.5 if n == 1 else j / (n - 1)
            ax.plot(res["iters"], res["mae"], color=_shade(base, t),
                    lw=1.9, label=res["label"])

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"MAE to true $p$:  $\frac{1}{NB}\sum_{b,i}\,|M_i(p_i)-p_i|$")
    ax.set_title("Convergence of agent confidence to the true probability vector")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "convergence_mae.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
