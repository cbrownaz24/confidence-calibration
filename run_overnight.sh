#!/usr/bin/env bash
# =============================================================================
# Overnight algorithm-search driver for Claude Code (headless / -p mode).
#
# Each iteration is a fresh, bounded `claude -p` run that reads RESEARCH_BRIEF.md,
# rebuilds context from LAB_NOTEBOOK.md + git log, tries ONE algorithm idea,
# screens it with search_eval.py, logs the result, and commits locally. Running
# many short iterations (rather than one long session) is what makes this robust:
# if one iteration derails, the next starts clean from the notebook + git state.
#
# Usage:
#   ./run_overnight.sh [ITERATIONS] [MODEL]
#   ./run_overnight.sh 15 sonnet
#
# BEFORE YOU SLEEP:
#   1. Make sure `claude` is installed and authenticated (`claude` once, interactively).
#   2. Set up the venv so the experiments run:  pip install -r requirements.txt
#   3. Commit your current work; this script runs on a dedicated branch and never
#      pushes, so your morning review is just `git log overnight/explore`.
#   4. Strongly consider running this in a throwaway clone or a container/VM: the
#      bypass permission mode below lets the agent run commands without asking.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

ITERS="${1:-12}"
MODEL="${2:-sonnet}"           # sonnet for cost in a loop; opus for harder reasoning
BRANCH="overnight/explore"
mkdir -p logs

# Work on a dedicated branch so the morning review is easy and main stays clean.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not a git repo — run 'git init' first"; exit 1; }
git checkout -B "$BRANCH"
git add -A && git commit -m "overnight: checkpoint before search" >/dev/null 2>&1 || true

echo "starting $ITERS iterations on branch $BRANCH with model $MODEL  ($(date))"

for i in $(seq 1 "$ITERS"); do
  echo "=== iteration $i / $ITERS  $(date) ==="
  PROMPT="$(cat RESEARCH_BRIEF.md)

You are iteration $i of $ITERS in an overnight search. Follow the per-iteration
protocol above exactly. Do ONE idea, screen it, log it to LAB_NOTEBOOK.md, update
tex/algorithms.tex, and commit locally. Then stop."

  # Permission model: --dangerously-skip-permissions runs unattended without
  # prompting. Claude Code is sandboxed to this directory by default, and this
  # script never pushes. For a tighter setup on a shared machine, replace the
  # bypass flag with:
  #   --permission-mode dontAsk \
  #   --allowedTools "Read" "Write" "Edit" "Glob" "Grep" "Bash(python:*)" "Bash(git:*)"
  # (adjust the Bash() pattern syntax to your Claude Code version).
  claude -p "$PROMPT" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --max-turns 60 \
    --output-format stream-json \
    >> "logs/iter_$(printf '%02d' "$i").jsonl" 2>> "logs/iter_$(printf '%02d' "$i").err" \
    || echo "  iteration $i exited non-zero (turn/budget limit or error) — continuing"

  echo "  done iteration $i"
done

echo
echo "FINISHED $(date). Review in the morning with:"
echo "  git checkout $BRANCH && git log --oneline"
echo "  cat LAB_NOTEBOOK.md"
echo "  python plot_experiments.py        # regenerate the full plot with new algos"
