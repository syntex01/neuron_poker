#!/usr/bin/env bash
# Cloud runner, started as root by cloud-init on a rented GPU box.
# Runs tests + benchmark, then trains until HOURS expires. Everything lands in
# /home/ubuntu/neuron_poker/results (world-readable, pickable via JupyterLab);
# when GH_TOKEN is set, results are also pushed to the repo branch every 20
# minutes. The instance is terminated from outside via the Lambda API.
set -x
BRANCH=claude/poker-sota-research-m9bqnn
HOURS=${HOURS:-4}
REPO=https://github.com/syntex01/neuron_poker
[ -n "$GH_TOKEN" ] && REPO="https://oauth2:${GH_TOKEN}@github.com/syntex01/neuron_poker"
cd /home/ubuntu
git clone -b "$BRANCH" "$REPO" neuron_poker
cd neuron_poker && mkdir -p results
git config user.email cloud-run@invalid && git config user.name cloud-run
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install -q -U pip && pip install -q -U "jax[cuda12]" pytest treys
{ python -c 'import jax; print("jax", jax.__version__, jax.devices())'
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python -m pytest poker_jax/tests -q
  python -m poker_jax.benchmark 8192 65536
} > results/run.log 2>&1
nohup python -m poker_jax.train 1000000 4096 128 > results/train.log 2>&1 &
TRAIN=$!

snapshot() {
  latest=$(ls -t poker_jax/ckpt 2>/dev/null | head -1)
  [ -n "$latest" ] && cp -f "poker_jax/ckpt/$latest" results/params_latest.pkl
  date -u +%FT%TZ > "results/$1"
  chmod -R a+r results
  [ -n "$GH_TOKEN" ] && git add results &&
    git commit -qm "cloud run: $1 $(date -u +%H%M)" && git push -q origin "$BRANCH"
}

END=$(( $(date +%s) + HOURS * 3600 ))
while kill -0 $TRAIN 2>/dev/null && [ "$(date +%s)" -lt $END ]; do
  sleep 600
  snapshot HEARTBEAT
done
kill $TRAIN 2>/dev/null; sleep 10
snapshot DONE
