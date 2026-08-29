# Handoff — 6-max NLHE SOTA Project (2026-08-29)

State of play for whoever picks this up (human or next Claude session).
Branch: `claude/poker-sota-research-m9bqnn`. All code lives in `poker_jax/`
(~1,060 LOC incl. tests); the legacy neuron_poker code at the repo root is
dead weight, kept only because this session lacked delete permission —
removing `agents/ gym_env/ tools/ tests/ main.py config.ini setup.py
requirements.txt .travis.yml .pylintrc .pydocstyle pytest.ini readme.rst`
is safe and desired (user wants minimal LOC).

## What exists and is verified

- `doc/SOTA_RESEARCH.md` — where poker AI stands (2026) and which lines
  transfer; `doc/BEATING_SOTA_PLAN.md` — the plan to surpass Pluribus-class
  play in 6-max NLHE.
- `poker_jax/engine.py` — vectorized 6-max NLHE (full no-limit rules: min
  raise, incomplete-all-in reopen rule, side/split pots). `cards.py` —
  7-card evaluator. Both validated: 10/10 tests in
  `poker_jax/tests/test_poker.py`, incl. 300k-pair ordering agreement with
  `treys` and 300 random hands matching an independent Python payout
  reference. ~145k env steps/s on a container CPU.
- `poker_jax/model.py` + `train.py` — self-play Trinal-Clip PPO, one shared
  net for all seats, hand-level credit assignment via backward scan;
  evaluation harness plays 3-copies-vs-3-copies with button rotation and
  reports bb/100 (`train.evaluate`, baselines `CALLER`/`RANDOM`).
- CPU sanity run (75 updates, batch 256): each snapshot beat the previous
  by +32 → +47 → +80 bb/100 (learning works); still loses to always-call
  (-300 bb/100) at that tiny scale — expected, entropy still ~max.

## In flight: first GPU run (not yet executed)

A ~$9, 4-hour A100 run on the user's Lambda account. The remote-session
sandbox could launch instances via the Lambda API but is (now) blocked from
doing so by its permission layer — and it can never SSH (HTTPS-443-only
egress) or transmit secrets. So the box must be started by the user:

1. Lambda dashboard → launch 1× A100 SXM4, us-east-1, any SSH key.
2. Open the instance's Cloud IDE (JupyterLab) → Terminal → paste:
   `curl -sL -o /tmp/run.sh https://raw.githubusercontent.com/syntex01/neuron_poker/ec6121ded1521795d23b3687c5efb3c9ede81d28/poker_jax/cloud_run.sh && nohup sudo GH_TOKEN=<fine-grained PAT, contents:write on this repo> HOURS=4 bash /tmp/run.sh > /tmp/poker_run.log 2>&1 &`
3. `cloud_run.sh` then: installs `jax[cuda12]`, runs tests + benchmark
   (`results/run.log`), trains 4h (`results/train.log`), commits
   `results/` (incl. `params_latest.pkl` and HEARTBEAT/DONE markers) to
   this branch every 20 min. Without GH_TOKEN it still runs; results then
   only live on the box (grab via JupyterLab before terminating).
4. **Lambda bills until the instance is TERMINATED** (not shutdown). The
   watcher must terminate via API or dashboard once `results/DONE` appears.
   Monitoring from a Claude session: poll the branch for results commits;
   terminate with the Lambda API (`POST /api/v1/instance-operations/terminate`).

Credentials: the user holds a Lambda API key and a repo-scoped fine-grained
GitHub PAT (created 2026-08-29, 7-day expiry). Both passed through chat and
should be treated as burned/revoked after the run. NEVER commit either.

## What the GPU results should show

- `run.log`: 10/10 tests on GPU + benchmark steps/s + hands/s (expect a
  large multiple of the 145k/16k CPU numbers).
- `train.log`: eval lines every 25 updates. Success = "vs snapshot"
  consistently positive AND "vs caller" trending up, ideally crossing 0
  (batch 4096 ⇒ ~0.5M transitions/update; 4h should reach hundreds of
  updates ≈ low-billions of transitions territory at good throughput).

## Next steps (priority order)

1. Analyze the GPU run; tune LR/entropy if vs-caller stalls.
2. Add action-history sequence encoding to `model.observe` (biggest known
   gap vs AlphaHoldem's card+action tensors; current obs aggregates bets).
3. K-best opponent pool in training (currently pure shared self-play).
4. Swap PPO objective for R-NaD-style regularization for 6-max convergence
   (see BEATING_SOTA_PLAN §3 Phase 1; OpenSpiel has reference R-NaD).
5. External HU yardsticks: Slumbot API, GTO Wizard Benchmark (AIVAT).
6. Later: depth-limited search with learned values (Phase 2), opponent
   modeling/exploitation (Phase 3).

Known trainer limitations (deliberate v1 cuts): transitions of hands left
unfinished at rollout end are dropped (no value bootstrap); no minibatching
(4 full-batch SGD steps); value target = final chips only. All fine at
smoke-test scale.
