# confidence-calibration

A modular simulation testbed for **truthful confidence elicitation among N agents**.

Each of `N` agents is handed its own private correctness probability `p_i`,
reports a confidence `c_i = M_i(p_i)` through its own MLP, and the realized
correctness is an exogenous draw `y_i ~ Bernoulli(p_i)`. A training reward (a
proper scoring rule, a second-price auction, a policy-gradient shaping, ...) is
supposed to make **truthful reporting `M_i(p) = p`** optimal. We compare reward
designs by how fast they drive every agent's report to the truth.

The setting is the one in `Alignment_Confidence_Notes.pdf`: sample
`(p_1, ..., p_N) ~ F`, give each agent its own `p_i`, draw `y_i ~ Bernoulli(p_i)`,
target `M_i(p) -> p`. The headline metric is the convergence of

```
MAE(t) = (1 / (B*N)) * sum_{b,i} | M_i(p_i) - p_i |
```

against the true N-dimensional probability vector `p`, over training steps.

## The workflow

The repo is built around one loop: **think of a new calibration algorithm → drop
one file in `src/` → run the plot script → see your algorithm in the legend.**
Every algorithm runs against the *same* environment, the same agent
architecture, the same training budget (all set in one config file), so the only
thing that differs between two curves on the plot is the algorithm itself.

```bash
pip install -r requirements.txt
python plot_experiments.py                 # runs every enabled algorithm, writes outputs/
python plot_experiments.py --config config/default.yaml
```

Outputs land in `outputs/`: `convergence_mae.png` (the headline plot) and
`results.json` (raw curves, so plots can be re-rendered without retraining).

## Layout

```
confidence-calibration/
  config/default.yaml      # THE config: environment, agents, training, toggles
  sim/                     # the constant simulation (stable API)
    environment.py         #   (p, z, y) generator from the Confidence Notes
    agents.py              #   N independent MLP agents (deterministic + stochastic bids)
    mechanism.py           #   second-price primitives + counterfactual loss
    updates.py             #   RLOO / GRPO / winner-mask advantage helpers
    metrics.py             #   the MAE-to-p headline metric
    algorithm.py           #   Algorithm base class + registry + train/eval loop
    plotting.py            #   plot registry + family palette
  src/                     # one file per algorithm (auto-discovered)
  plots/                   # one file per plot type (auto-discovered)
  plot_experiments.py      # discover -> filter -> run -> render
```

`sim/` is the part meant to stay constant. `src/` and `plots/` are where the repo
grows.

## Adding an algorithm

Create a file in `src/`. Subclass `Algorithm`, set `name`/`family`, and implement
`compute_loss(batch, ctx)` where `batch = (p, z, y)` (each `[B, N]`) and `ctx`
carries the annealed schedules (`ctx.tau`, `ctx.sigma`) and an rng. Return a
scalar loss; the base class handles per-agent optimizers, evaluation, the
convergence curve, and registration.

```python
# src/score_brier.py
from sim import Algorithm, register_algorithm

@register_algorithm
class Brier(Algorithm):
    name = "brier"            # unique id + legend key
    family = "score"          # color group on the plot

    def compute_loss(self, batch, ctx):
        p, z, y = batch
        b = self.ens.bids(z)               # deterministic [B, N] reports
        return ((b - y) ** 2).mean(0).sum()
```

For policy-gradient algorithms, call `self.ens.stochastic_bids(z, ctx.sigma)` to
get sampled bids and their log-probs, build a per-agent reward, turn it into an
advantage (`rloo_advantage`, `grpo_advantage`, or `self.ema_advantage`), and
return `-(adv * logp).mean()`. For auction algorithms, form the competing price
`s` and call `counterfactual_loss(b, s, y, ctx.tau)`. See the existing files for
worked examples of each pattern.

## Adding a plot

Drop a file in `plots/` with a `@register_plot("name")` function taking
`(results, out_dir, cfg)` and returning the path it wrote. `results` is the list
of per-algorithm dicts (`name`, `label`, `family`, `iters`, `mae`,
`final_mae`, `per_agent_mae`).

## Toggling what runs

Everything in `src/` and `plots/` is discovered automatically; the config decides
what actually runs:

```yaml
algorithms:
  enabled: all          # or an explicit list of names
  disabled: [grpo_brier]
plots:
  enabled: all
```

## The algorithms (current set)

Ported from `auction-confidence`, `alignment-auction`, `spauction-rafael`, and
the two GRPO / multi-agent notebooks. Family = color group on the plot.

| family | files | idea |
|---|---|---|
| `score` | `score_brier`, `score_log`, `score_spherical`, `score_supervised` | strictly proper scoring rules (differentiable, dense gradient); `supervised` uses latent `p` as an oracle ceiling |
| `counterfactual` | `cf_second_price`, `cf_reserve`, `cf_pop_reserve`, `cf_random_price`, `cf_sleep` | differentiable soft second-price auctions; reserves restore the proper-scoring optimum, `random_price` reduces to Brier |
| `reinforce` | `pg_winner_only`, `pg_losers_zero`, `pg_first_price`, `pg_rloo_auction`, `pg_rloo_log`, `pg_regret_surrogate` | score-function policy gradient; winner-only collapses, RLOO/regret-surrogate fix it, `rloo_log` sidesteps the auction |
| `grpo` | `grpo_second_price`, `grpo_brier` | group-relative standardized advantage over the N agents |
```
