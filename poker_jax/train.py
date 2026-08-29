"""Self-play Trinal-Clip PPO on the 6-max engine (AlphaHoldem-style).

One shared network plays all six seats; each action is credited with its
seat's net chips for the hand (undiscounted, assigned by a backward scan over
the terminal rewards). The policy-loss ratio is additionally clipped from
above for negative advantages (Trinal-Clip PPO), which tames the huge
variance of all-in spots.

Usage: python -m poker_jax.train [updates] [batch] [horizon]
Checkpoints are pickled to poker_jax/ckpt/; evaluation prints bb/100 (per
seat, 3 copies vs 3 copies) against an always-call bot and the previous
snapshot.
"""

import functools
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp

import poker_jax as pj
from poker_jax.model import (
    REWARD_SCALE, adam_init, adam_step, net_apply, net_init, observe,
)

CLIP, DELTA, VF_COEF, ENT_COEF, LR, SGD_STEPS = 0.2, 3.0, 0.5, 0.01, 3e-4, 4
NEG_INF = -1e9


def _masked(logits, mask):
    return jnp.where(mask, logits, NEG_INF)


def rollout(params, states, key, horizon):
    """Run `horizon` self-play steps; returns (states, flat transition batch)."""

    def one_step(carry, _):
        states, key = carry
        key, k = jax.random.split(key)
        obs = jax.vmap(observe)(states)
        logits, value = net_apply(params, obs)
        mask = jax.vmap(pj.legal_action_mask)(states)
        logp_all = jax.nn.log_softmax(_masked(logits, mask))
        action = jax.random.categorical(k, logp_all)
        seat = states.current
        states, rewards, terminal = jax.vmap(pj.step_autoreset)(states, action)
        logp = jnp.take_along_axis(logp_all, action[:, None], 1)[:, 0]
        return (states, key), dict(
            obs=obs, mask=mask, action=action, logp=logp, value=value,
            seat=seat, rewards=rewards, terminal=terminal,
        )

    (states, _), traj = jax.lax.scan(one_step, (states, key), None, length=horizon)

    def assign_returns(carry, x):  # backward: propagate each hand's payout
        ret, valid = carry
        ret = jnp.where(x["terminal"][:, None], x["rewards"], ret)
        valid = x["terminal"] | valid
        return (ret, valid), (ret, valid)

    batch = traj["rewards"].shape[1]
    (_, (rets, valid)) = jax.lax.scan(
        assign_returns,
        (jnp.zeros((batch, pj.NUM_SEATS), jnp.int32), jnp.zeros(batch, bool)),
        traj, reverse=True,
    )
    ret = jnp.take_along_axis(rets, traj["seat"][..., None], 2)[..., 0] / REWARD_SCALE
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    return states, {
        "obs": flat(traj["obs"]), "mask": flat(traj["mask"]),
        "action": flat(traj["action"]), "logp": flat(traj["logp"]),
        "value": flat(traj["value"]), "ret": flat(ret), "w": flat(valid),
    }


def update(params, opt, batch):
    w = batch["w"] / batch["w"].sum()
    wmean = lambda x: (x * w).sum()
    adv = batch["ret"] - batch["value"]
    adv = (adv - wmean(adv)) / (jnp.sqrt(wmean((adv - wmean(adv)) ** 2)) + 1e-8)

    def loss_fn(p):
        logits, value = net_apply(p, batch["obs"])
        logp_all = jax.nn.log_softmax(_masked(logits, batch["mask"]))
        logp = jnp.take_along_axis(logp_all, batch["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - batch["logp"])
        ratio = jnp.where(adv < 0, jnp.minimum(ratio, DELTA), ratio)
        pg = -wmean(
            jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv)
        )
        v_loss = wmean((value - batch["ret"]) ** 2)
        ent = wmean(-(jnp.exp(logp_all) * logp_all * batch["mask"]).sum(-1))
        return pg + VF_COEF * v_loss - ENT_COEF * ent, (pg, v_loss, ent)

    for _ in range(SGD_STEPS):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        params, opt = adam_step(params, grads, opt, LR)
    return params, opt, jnp.stack([loss, *aux])


def _net_policy(params, obs):
    return net_apply(params, obs)[0]


def policy(params):
    """A policy is (fn, params) with fn(params, obs) -> logits."""
    return (_net_policy, params)


CALLER = (lambda p, obs: jnp.where(jnp.arange(pj.NUM_ACTIONS) == 1, 0.0, NEG_INF), ())
RANDOM = (lambda p, obs: jnp.zeros(pj.NUM_ACTIONS), ())


@functools.partial(jax.jit, static_argnums=(0, 1))
def _eval_block(fn_a, fn_b, params_a, params_b, states, key):
    def one_step(carry, _):
        states, key = carry
        key, k = jax.random.split(key)
        obs = jax.vmap(observe)(states)
        mask = jax.vmap(pj.legal_action_mask)(states)
        logits = jnp.where(
            (states.current % 2 == 0)[:, None],
            fn_a(params_a, obs), fn_b(params_b, obs),
        )
        action = jax.random.categorical(k, _masked(logits, mask))
        states, rewards, terminal = jax.vmap(pj.step_autoreset)(states, action)
        return (states, key), (rewards[:, ::2].sum(), terminal.sum())

    (states, key), (chips, hands) = jax.lax.scan(
        one_step, (states, key), None, length=256
    )
    return states, key, chips.sum(), hands.sum()


def evaluate(policy_a, policy_b, key, batch=512, min_hands=30_000):
    """bb/100 earned per seat by policy_a (even seats) vs policy_b (odd seats)."""
    (fn_a, params_a), (fn_b, params_b) = policy_a, policy_b
    keys = jax.random.split(key, batch)
    states = jax.vmap(pj.init, in_axes=(0, None))(keys, 0)
    chips = hands = 0
    while hands < min_hands:
        states, key, c, h = _eval_block(fn_a, fn_b, params_a, params_b, states, key)
        chips, hands = chips + int(c), hands + int(h)
    return chips / pj.BIG_BLIND / (3 * hands) * 100


def main(updates=200, batch=512, horizon=128):
    ckpt_dir = pathlib.Path(__file__).parent / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    key = jax.random.PRNGKey(0)
    params = net_init(key)
    opt = adam_init(params)
    states = jax.vmap(pj.init, in_axes=(0, None))(
        jax.random.split(key, batch), 0
    )
    snapshot = jax.tree_util.tree_map(jnp.copy, params)
    roll = jax.jit(rollout, static_argnums=3)
    upd = jax.jit(update)

    for u in range(1, updates + 1):
        key, k = jax.random.split(key)
        states, transitions = roll(params, states, k, horizon)
        params, opt, stats = upd(params, opt, transitions)
        if u % 25 == 0 or u == updates:
            key, k1, k2 = jax.random.split(key, 3)
            vs_call = evaluate(policy(params), CALLER, k1)
            vs_snap = evaluate(policy(params), policy(snapshot), k2)
            loss, pg, vl, ent = [float(x) for x in stats]
            print(
                f"update {u:>4}  loss {loss:+.3f} pg {pg:+.3f} v {vl:.3f} "
                f"ent {ent:.2f}  bb/100 vs caller {vs_call:+8.1f} "
                f"vs snapshot {vs_snap:+8.1f}", flush=True,
            )
            snapshot = jax.tree_util.tree_map(jnp.copy, params)
            (ckpt_dir / f"params_{u}.pkl").write_bytes(pickle.dumps(params))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
