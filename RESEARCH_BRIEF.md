# RESEARCH BRIEF — find an auction rule that competes with brier/log

## Goal

Find an **auction-family** confidence-calibration rule whose MAE-to-`p` convergence
is competitive with the proper scoring rules `brier` and `log` — ideally matching
them on `env.kind=uniform` AND staying strong on `correlated` (the winner's-curse
regime) and `beta`. "Competitive" = seed-averaged final MAE within ~1.2x of `log`
on `uniform`, and clearly better than `cf_second_price` on `correlated`.

A working baseline already exists: `cf_pairwise_roundrobin` (dense pairwise
auctions) ties `brier` on `uniform`. Your job is to push further — beat it, make it
robust across `env.kind`, or find a different auction mechanism that does.

## Leads worth exploring (pick ONE per iteration)

1. **Loser starvation.** In the N-way auction, losers sit where peers never price
   them, so their counterfactual gradient `sigma((b_i-s_i)/tau)*(y_i-s_i)` is tiny.
   Densify the signal so every agent learns every round.
2. **Pairwise round-robin (already seeded).** Variants: weighted matchups, a reserve
   inside each pairwise contest, pooling the pairwise prices, or a *realized*
   (REINFORCE) pairwise tournament with an RLOO baseline across opponents.
3. **Loser counterfactual surplus.** Give losers their would-have-won surplus
   `min/max(p_i - s_i, 0)`-style signal (cf. `pg_regret_surrogate`, exp3/exp4).
4. **Soft / all-pay allocation.** A softmax allocation so all agents get partial,
   differentiable credit rather than a single hard winner.
5. **Reserve design.** Reserve drawn from the peer-bid distribution vs uniform vs
   annealed; this is what restores the independent full-support competitor.
6. **Correlated-values robustness.** Whatever you try, check it on
   `env.kind=correlated` — that is where the auction is supposed to struggle and
   where a real win would be most interesting.

## Per-iteration protocol (you start fresh each iteration — rebuild context first)

1. **Read state.** Read `LAB_NOTEBOOK.md` and run `git log --oneline -20` to see what
   has already been tried and what worked. Read `CLAUDE.md` for repo conventions.
2. **One hypothesis.** State a single concrete idea and why it should help.
3. **Implement.** Add ONE new file to `src/` following the blueprint. Do not touch
   `sim/`, existing `src/` files, or `config/default.yaml`.
4. **Screen.** `python search_eval.py --seeds 3 --algos log brier cf_second_price
   cf_pairwise_roundrobin <your_new_name>`. Compare the seed-averaged MAE.
5. **If promising**, validate harder: re-run with `--seeds 5`, and screen on the
   correlated regime (make a `config/search_correlated.yaml` with `env.kind:
   correlated`, everything else identical) and on the full budget via
   `config/default.yaml`. A finalist must hold up across seeds and ideally across
   `env.kind`.
6. **Log.** Append a dated entry to `LAB_NOTEBOOK.md`: hypothesis, the exact MAE
   numbers (mean±std vs the baselines), verdict (win / tie / fail), and the next
   idea it suggests. Be concise and honest — record failures too; they prune the
   search.
7. **Document.** Add one row for the new algorithm to the right family block in
   `tex/algorithms.tex`, same minimal style as existing rows.
8. **Commit.** `git add -A && git commit -m "explore: <short description> (MAE
   x.xxx)"`. Commit locally only — never push.

## Success criteria for the night

A short list (in `LAB_NOTEBOOK.md`) of auction rules that match or beat `brier`/`log`
on `uniform`, with at least one candidate that also holds up on `correlated`, each
with a one-line LaTeX rule in `tex/algorithms.tex` and a reproducible
`search_eval.py` command. If nothing beats the pairwise baseline, that is still a
useful result — say so clearly and characterize *why* the auction is hard to beat.

## Guardrails

- Local commits only; no pushes, no force, no deletes, no network beyond what the
  experiments need.
- Bounded experiments: use `config/search.yaml` (1200 steps) for screening; reserve
  the full budget for finalists.
- Keep the simulation constant: never edit env/agent fields so curves stay
  comparable. New screening configs are fine as long as the env/agent blocks match
  `default.yaml` (only `env.kind` may change, to probe robustness).
