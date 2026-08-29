"""Observation encoding, policy/value MLP, and Adam — pure JAX, no deps.

The observation is seat-relative (the acting seat is always slot 0), so one
network plays every seat: 52 hole-card bits, 52 revealed-board bits, per-seat
chip/status features, street, pot odds context, and button position. Betting
is summarized per seat per hand (AlphaHoldem-style full action-history
tensors are the planned upgrade).
"""

import jax
import jax.numpy as jnp

from poker_jax.engine import BIG_BLIND, NUM_ACTIONS, NUM_SEATS, STARTING_STACK

_SEATS = jnp.arange(NUM_SEATS)
_REVEALED = jnp.array([0, 3, 4, 5])  # board cards visible per street
OBS_DIM = 52 + 52 + 6 * NUM_SEATS + 4 + 3 + NUM_SEATS
REWARD_SCALE = float(STARTING_STACK)


def observe(state):
    """State -> (OBS_DIM,) float32 features for the acting seat."""
    cur = state.current
    order = (cur + _SEATS) % NUM_SEATS  # seats in relative order, self first
    stack_f = 1.0 / STARTING_STACK
    hole = jax.nn.one_hot(state.hole[cur], 52).sum(0)
    board = (
        jax.nn.one_hot(state.board, 52)
        * (jnp.arange(5) < _REVEALED[state.street])[:, None]
    ).sum(0)
    seats = jnp.stack(
        [
            state.stacks[order] * stack_f,
            state.street_contrib[order] * stack_f,
            state.total_contrib[order] * stack_f,
            state.folded[order],
            state.all_in[order],
            state.acted[order],
        ],
        axis=1,
    ).reshape(-1)
    scalars = jnp.array(
        [
            state.total_contrib.sum() * stack_f / NUM_SEATS,
            (state.max_bet - state.street_contrib[cur]) * stack_f,
            state.min_raise_inc * stack_f,
        ]
    )
    street = jnp.arange(4) == state.street
    button = _SEATS == (state.button - cur) % NUM_SEATS
    return jnp.concatenate([hole, board, seats, scalars, street, button]).astype(
        jnp.float32
    )


def net_init(key, hidden=(512, 512)):
    sizes = (OBS_DIM,) + hidden
    keys = jax.random.split(key, len(hidden) + 2)
    def dense(k, n_in, n_out):
        return (
            jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in),
            jnp.zeros(n_out),
        )
    return {
        "trunk": [dense(k, sizes[i], sizes[i + 1]) for i, k in enumerate(keys[:-2])],
        "pi": dense(keys[-2], sizes[-1], NUM_ACTIONS),
        "v": dense(keys[-1], sizes[-1], 1),
    }


def net_apply(params, obs):
    """obs (..., OBS_DIM) -> (logits (..., NUM_ACTIONS), value (...,))."""
    x = obs
    for w, b in params["trunk"]:
        x = jax.nn.relu(x @ w + b)
    logits = x @ params["pi"][0] + params["pi"][1]
    value = (x @ params["v"][0] + params["v"][1])[..., 0]
    return logits * 0.01, value


def adam_init(params):
    zeros = lambda: jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros(), "v": zeros(), "t": jnp.zeros((), jnp.int32)}


def adam_step(params, grads, opt, lr, max_norm=1.0, b1=0.9, b2=0.999, eps=1e-8):
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    grads = jax.tree_util.tree_map(
        lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-9)), grads
    )
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    scale = lr * jnp.sqrt(1 - b2**t) / (1 - b1**t)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - scale * m / (jnp.sqrt(v) + eps), params, m, v
    )
    return params, {"m": m, "v": v, "t": t}
