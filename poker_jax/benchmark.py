"""Throughput benchmark: random-policy self-play, fully jitted on device.

Usage: python -m poker_jax.benchmark [batch ...]
"""

import sys
import time

import jax
import jax.numpy as jnp

import poker_jax as pj


def run(batch, steps):
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    states = jax.vmap(pj.init, in_axes=(0, None))(keys, 0)

    @jax.jit
    def rollout(states, key):
        def body(_, carry):
            states, key, hands = carry
            key, k = jax.random.split(key)
            masks = jax.vmap(pj.legal_action_mask)(states)
            actions = jax.random.categorical(k, jnp.where(masks, 0.0, -1e9))
            states, _, terminal = jax.vmap(pj.step_autoreset)(states, actions)
            return states, key, hands + terminal.sum()
        return jax.lax.fori_loop(0, steps, body, (states, key, jnp.int32(0)))

    key = jax.random.PRNGKey(1)
    rollout(states, key)[2].block_until_ready()  # compile
    t0 = time.perf_counter()
    hands = int(rollout(states, key)[2].block_until_ready())
    dt = time.perf_counter() - t0
    print(
        f"batch {batch:>6}: {batch * steps / dt:>12,.0f} steps/s "
        f"{hands / dt:>12,.0f} hands/s  ({jax.devices()[0].platform})"
    )


if __name__ == "__main__":
    batches = [int(b) for b in sys.argv[1:]] or [1024, 8192]
    for b in batches:
        run(b, steps=500)
