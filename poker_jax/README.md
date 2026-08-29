# poker_jax

Vectorized 6-max No-Limit Hold'em in pure JAX — Phase 0 of
[doc/BEATING_SOTA_PLAN.md](../doc/BEATING_SOTA_PLAN.md). One `init`/`step`
state machine (full NL rules: min-raise, incomplete-all-in reopen rule, side
pots, split pots) plus a branch-free 7-card evaluator. Everything jits and
vmaps, so the same code runs batched on CPU or GPU.

```bash
pip install -r poker_jax/requirements.txt          # CPU
pip install -U "jax[cuda12]"                       # NVIDIA GPU (e.g. RTX 3070)

python -m pytest poker_jax/tests -q
python -m poker_jax.benchmark 8192 32768
python -m poker_jax.train                # or: train <updates> <batch> <horizon>
```

`train.py` runs self-play Trinal-Clip PPO (AlphaHoldem-style) with one shared
network on all six seats, prints bb/100 vs an always-call bot and vs the
previous snapshot every 25 updates, and pickles checkpoints to
`poker_jax/ckpt/`. On a GPU raise the defaults, e.g.
`python -m poker_jax.train 5000 4096 128`.

API: `init(key, button) -> State`, `step(state, action) -> State`,
`legal_action_mask(state)`, `step_autoreset(state, action)` for training
loops; vmap them over a batch of states. Actions: fold, check/call, six
pot-fraction raises, all-in. Rewards are net chips on the terminal step
(BB = 2 chips, 100bb stacks).
