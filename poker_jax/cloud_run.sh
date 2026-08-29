#!/usr/bin/env bash
# Self-contained cloud runner, started by cloud-init on a rented GPU box.
# Env: GH_TOKEN (repo-scoped, contents read/write), HOURS (wall-clock budget).
# Installs deps, runs tests + benchmark, trains until the budget expires, and
# pushes logs + the latest checkpoint to the repo branch every 20 minutes.
set -x
BRANCH=claude/poker-sota-research-m9bqnn
HOURS=${HOURS:-3}
cd /root
git clone -b "$BRANCH" "https://oauth2:${GH_TOKEN}@github.com/syntex01/neuron_poker" repo
cd repo
git config user.email cloud-run@invalid && git config user.name cloud-run
python3 -m venv venv && . venv/bin/activate
pip install -q -U pip && pip install -q -U "jax[cuda12]" pytest treys
mkdir -p results
python -c 'import jax; print(jax.__version__, jax.devices())' > results/run.log 2>&1
python -m pytest poker_jax/tests -q >> results/run.log 2>&1
python -m poker_jax.benchmark 8192 65536 >> results/run.log 2>&1
nohup python -m poker_jax.train 1000000 4096 128 > results/train.log 2>&1 &
TRAIN=$!

push_results() {
  latest=$(ls -t poker_jax/ckpt 2>/dev/null | head -1)
  [ -n "$latest" ] && cp -f "poker_jax/ckpt/$latest" results/params_latest.pkl
  nvidia-smi --query-gpu=name,utilization.gpu,power.draw --format=csv \
    > results/gpu.txt 2>&1
  git add results && git commit -qm "cloud run: $1 $(date -u +%FT%TZ)" \
    && git push -q origin "$BRANCH"
}

END=$(( $(date +%s) + HOURS * 3600 ))
while kill -0 $TRAIN 2>/dev/null && [ "$(date +%s)" -lt $END ]; do
  sleep 1200
  push_results progress
done
kill $TRAIN 2>/dev/null; sleep 10
push_results final
