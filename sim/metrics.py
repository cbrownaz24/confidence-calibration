"""Evaluation metrics. The headline quantity is the truthfulness MAE."""

from __future__ import annotations

import torch


@torch.no_grad()
def mae_to_p(ensemble, z_eval: torch.Tensor, p_eval: torch.Tensor) -> float:
    """Mean absolute error of the deterministic reports against the true p.

        MAE = (1 / (B*N)) sum_{b,i} | M_i(z_i) - p_i |

    This is the noise-free convergence metric the notes ask for: we *know* the
    true N-dimensional probability vector p, so distance-to-p measures exactly
    how close each agent's reporting map is to the truthful identity.
    """
    bids = ensemble.bids(z_eval)
    return float((bids - p_eval).abs().mean())


@torch.no_grad()
def per_agent_mae(ensemble, z_eval: torch.Tensor, p_eval: torch.Tensor) -> list:
    bids = ensemble.bids(z_eval)
    return (bids - p_eval).abs().mean(dim=0).cpu().tolist()


# --------------------------------------------------------------------------- #
# Calibration metrics. These compare reported confidence against *binary*
# outcomes y (not the hidden p), so they measure exactly what is observable in
# the hidden-features setting: is a reported confidence of 0.7 right 70% of the
# time? MAE-to-p measures "wrong vs right"; ECE measures "miscalibrated vs
# calibrated" -- the two are different failure modes.

def _bin_stats(conf: torch.Tensor, correct: torch.Tensor, n_bins: int):
    """Equal-width reliability bins over [0,1]; returns per-bin tensors."""
    conf = conf.reshape(-1)
    correct = correct.reshape(-1).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; right-closed except the first bin includes 0.
    idx = torch.bucketize(conf, edges[1:-1].contiguous(), right=False)
    out = []
    for k in range(n_bins):
        m = idx == k
        cnt = int(m.sum())
        if cnt == 0:
            out.append((float("nan"), float("nan"), 0))
        else:
            out.append((float(conf[m].mean()), float(correct[m].mean()), cnt))
    return out


@torch.no_grad()
def ece(ensemble, z_eval: torch.Tensor, y_eval: torch.Tensor,
        n_bins: int = 15) -> float:
    """Expected Calibration Error of the reported confidences against outcomes.

        ECE = sum_k (n_k / N) * | acc_k - conf_k |

    aggregated over all agents (each (agent, item) pair is one prediction).
    """
    bids = ensemble.bids(z_eval)
    total = bids.numel()
    stats = _bin_stats(bids, y_eval, n_bins)
    e = 0.0
    for conf_k, acc_k, cnt in stats:
        if cnt > 0:
            e += (cnt / total) * abs(acc_k - conf_k)
    return float(e)


@torch.no_grad()
def reliability_curve(ensemble, z_eval: torch.Tensor, y_eval: torch.Tensor,
                      n_bins: int = 15) -> list:
    """Per-bin (mean confidence, empirical accuracy, fraction of mass).

    Returned as plain Python lists so it round-trips through results.json. Empty
    bins are dropped. Feeds the reliability-diagram plot.
    """
    bids = ensemble.bids(z_eval)
    total = bids.numel()
    out = []
    for conf_k, acc_k, cnt in _bin_stats(bids, y_eval, n_bins):
        if cnt > 0:
            out.append([conf_k, acc_k, cnt / total])
    return out
