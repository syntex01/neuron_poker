"""GPU-native, fully vectorized 6-max No-Limit Texas Hold'em engine in JAX.

Phase 0 of the plan in doc/BEATING_SOTA_PLAN.md: a simulator that can run
millions of hands per second batched on a GPU, as the substrate for
large-scale self-play training.
"""

from poker_jax.cards import hand_score
from poker_jax.engine import (
    BIG_BLIND,
    NUM_ACTIONS,
    NUM_SEATS,
    SMALL_BLIND,
    STARTING_STACK,
    State,
    init,
    legal_action_mask,
    step,
    step_autoreset,
)

__all__ = [
    "BIG_BLIND",
    "NUM_ACTIONS",
    "NUM_SEATS",
    "SMALL_BLIND",
    "STARTING_STACK",
    "State",
    "hand_score",
    "init",
    "legal_action_mask",
    "step",
    "step_autoreset",
]
