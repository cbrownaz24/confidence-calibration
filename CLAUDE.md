# CLAUDE.md — working notes for agents in this repo

This repo studies **truthful confidence elicitation among N agents**. N independent
MLPs each get a private correctness probability `p_i`, report `c_i = M_i(p_i)`, and
see an exogenous outcome `y_i ~ Bernoulli(p_i)`. A reward/loss rule is meant to make
truthful reporting `M_i(p)=p` optimal. We compare rules by how fast the MAE-to-`p`
converges. Read `README.md` first; the setting is `Alignment_Confidence_Notes.pdf`.

## The blueprint (how to add an algorithm)

One algorithm = one file in `src/`. Subclass `Algorithm` (in `sim/algorithm.py`),
set `name`/`family`/`label`, implement **only** `compute_loss(self, batch, ctx)`
where `batch=(p, z, y)` each `[B,N]` and `ctx` carries `ctx.tau`, `ctx.sigma`,
`ctx.eps_reserve`, `ctx.eps_sleep`, `ctx.gen`. Return a scalar loss. The base class
handles per-agent optimizers, schedules, evaluation, the convergence curve, and
registration via `@register_algorithm`. Everything is auto-discovered.

Patterns (copy an existing file):
- **differentiable score**: use `self.ens.bids(z)`; return a pathwise loss.
- **counterfactual auction**: `b=self.ens.bids(z)`; form a detached price `s`; call
  `counterfactual_loss(b, s, y, ctx.tau)`. See `cf_pairwise_roundrobin.py` for the
  dense-pairwise pattern.
- **policy gradient**: `b, logp = self.ens.stochastic_bids(z, ctx.sigma)`; build a
  per-agent reward; turn it into an advantage (`rloo_advantage`, `grpo_advantage`,
  or `self.ema_advantage(r, ctx)`); return `-(adv * logp).mean()`.

`family` must be one of `score | counterfactual | reinforce | grpo` (controls the
plot color). The public API of `sim/` is in `sim/__init__.py`.

## Commands

```bash
python search_eval.py --seeds 3 --algos <names...>   # fast multi-seed screen (config/search.yaml)
python search_eval.py --seeds 3                       # screen ALL registered algorithms
python plot_experiments.py                            # full run + outputs/convergence_mae.png
python plot_experiments.py --config config/search.yaml
```

`search_eval.py` is the judge: it reports mean±std final MAE over seeds. **Always
compare on the seed-averaged number** — the policy-gradient curves are noisy and a
single seed is not trustworthy.

## Hard rules (do NOT break)

- Do **not** change the public API of `sim/`, the existing `src/` algorithms, or
  the env/agent fields in `config/default.yaml`. The simulation is held constant so
  curves are comparable. (You may add a new `config/*.yaml` for your own screening.)
- Do **not** delete files, force-push, or push to any remote. Commit locally only.
- Keep every new algorithm faithful to the shared loop — do not special-case the
  training loop inside a `src/` file.
- For every new algorithm, add one row to the matching family block in
  `tex/algorithms.tex`, in the same minimal style as the existing rows.
