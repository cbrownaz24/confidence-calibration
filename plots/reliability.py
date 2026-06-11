"""Reliability diagrams -- the picture of calibration on held-out questions.

For each algorithm we plot empirical accuracy against mean reported confidence,
per confidence bin, evaluated on held-out questions. A perfectly calibrated
reporter lies on the diagonal; bins below the diagonal are overconfident (claimed
more than it delivered), bins above are underconfident.

Because the probability is never revealed in the hidden-features environment,
this is calibration on *novel inputs* -- the phenomenon of interest, not the
trivial training-set version. One small panel per algorithm; marker size encodes
how much probability mass landed in each bin.
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import register_plot, FAMILY_COLORS


@register_plot("reliability")
def plot(results: list[dict], out_dir: str, cfg=None) -> str:
    panels = [r for r in results if r.get("reliability")]
    if not panels:
        # nothing to draw (e.g. a run without calibration data); emit a stub.
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "no reliability data", ha="center", va="center")
        ax.axis("off")
        path = os.path.join(out_dir, "reliability.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    n = len(panels)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow),
                             squeeze=False)

    for k, res in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        color = FAMILY_COLORS.get(res["family"], FAMILY_COLORS["misc"])
        conf = [b[0] for b in res["reliability"]]
        acc = [b[1] for b in res["reliability"]]
        mass = [b[2] for b in res["reliability"]]
        sizes = [20.0 + 380.0 * m for m in mass]
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.6", zorder=1)
        ax.plot(conf, acc, "-", lw=1.2, color=color, alpha=0.7, zorder=2)
        ax.scatter(conf, acc, s=sizes, color=color, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{res['label']}\nECE={res.get('final_ece', float('nan')):.3f}",
                     fontsize=9)
        ax.grid(True, alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel("empirical accuracy")
        if k // ncol == nrow - 1:
            ax.set_xlabel("reported confidence")

    # hide any unused panels
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Reliability on held-out questions "
                 "(below diagonal = overconfident)", y=1.0)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reliability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
