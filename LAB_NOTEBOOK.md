# Lab notebook — auction algorithm search

Append one entry per experiment. Keep it concise and honest; record failures too.

---

## Baseline standings (seed, before the overnight search)

`python search_eval.py --seeds 2` on `config/search.yaml` (uniform, N=5, 1200 steps),
seed-averaged final MAE-to-p, best first:

| algo | MAE (mean) | note |
|---|---|---|
| supervised | 0.005 | oracle ceiling (uses latent p) |
| log | 0.026 | proper-score target to beat |
| brier | 0.030 | proper-score target to beat |
| **cf_pairwise_roundrobin** | **0.031** | dense pairwise auction — ties brier |
| cf_pop_reserve | ~0.050 | best of the reserve auctions |
| cf_second_price | 0.070 | bare auction (the one to beat by densifying) |
| pg_* / grpo_* | 0.14–0.40 | policy-gradient, noisier |

**Read:** the dense pairwise auction already reaches the proper-score band on
`uniform`. Open questions: (a) can a different auction design *beat* log/brier, not
just tie? (b) does pairwise round-robin hold up on `correlated`? (c) is there a
realized (REINFORCE) auction that gets dense enough to compete?

---

<!-- New entries below. Format:
## YYYY-MM-DD HH:MM  <algo_name>
Hypothesis: ...
Command: python search_eval.py --seeds 3 --algos log brier cf_pairwise_roundrobin <name>
Result: <name> MAE x.xxx ± x.xxx  (log 0.026, brier 0.030)
Verdict: win / tie / fail — because ...
Next: ...
-->
