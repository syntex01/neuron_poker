"""Tests for the JAX NLHE engine: evaluator fuzz vs `treys`, betting-rule
scenarios, and a full-game fuzz against a plain-Python payout reference."""

import itertools
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import poker_jax as pj
from poker_jax.cards import hand_score

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "shdc"


def card(text):
    return RANK_CHARS.index(text[0]) * 4 + SUIT_CHARS.index(text[1])


def score(*cards_text):
    return int(hand_score(jnp.array([card(c) for c in cards_text], jnp.int32)))


# ---------------------------------------------------------------- evaluator


def test_known_hands():
    ordered = [
        score("As", "Kd", "Qh", "Jc", "9s", "7d", "5h"),  # high card
        score("As", "Ad", "Qh", "Jc", "9s", "7d", "5h"),  # pair
        score("As", "Ad", "Qh", "Qc", "9s", "7d", "5h"),  # two pair
        score("As", "Ad", "Ah", "Qc", "9s", "7d", "5h"),  # trips
        score("As", "2d", "3h", "4c", "5s", "9d", "Jh"),  # wheel straight
        score("2d", "3h", "4c", "5s", "6d", "9d", "Jh"),  # 6-high straight
        score("As", "Ks", "Qs", "Js", "8s", "2d", "3h"),  # flush
        score("As", "Ks", "Qs", "Js", "9s", "2d", "3h"),  # better flush kicker
        score("Ks", "Kd", "Kh", "Ac", "As", "5d", "6h"),  # kings full of aces
        score("As", "Ad", "Ah", "Kc", "Ks", "Kd", "5h"),  # two trips = aces full
        score("As", "Ad", "Ah", "Ac", "9s", "7d", "5h"),  # quads
        score("As", "2s", "3s", "4s", "5s", "Kd", "Kh"),  # steel wheel
        score("As", "Ks", "Qs", "Js", "Ts", "2d", "3h"),  # royal flush
    ]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered)
    # three pairs: best five is A A K K + Q kicker, the jack never plays
    assert score("As", "Ad", "Kh", "Kc", "Qs", "Qd", "Jh") == score(
        "As", "Ad", "Kh", "Kc", "Qs", "2d", "3h"
    )
    # board plays: identical scores
    assert score("2s", "3d", "As", "Ks", "Qs", "Js", "Ts") == score(
        "7h", "8c", "As", "Ks", "Qs", "Js", "Ts"
    )


def test_evaluator_fuzz_vs_treys():
    treys = pytest.importorskip("treys")
    evaluator = treys.Evaluator()
    rng = random.Random(0)
    deck = list(range(52))
    hands = []
    for _ in range(3000):
        rng.shuffle(deck)
        hands.append(deck[:7].copy())
    ours = np.asarray(hand_score(jnp.array(hands, jnp.int32)))
    theirs = np.array([
        evaluator.evaluate(
            [treys.Card.new(RANK_CHARS[c // 4] + SUIT_CHARS[c % 4]) for c in h[:2]],
            [treys.Card.new(RANK_CHARS[c // 4] + SUIT_CHARS[c % 4]) for c in h[2:]],
        )
        for h in hands
    ])
    # treys: lower is better. The two total orders must agree exactly.
    for i, j in itertools.islice(itertools.combinations(range(len(hands)), 2), 300000):
        assert np.sign(ours[i] - ours[j]) == np.sign(theirs[j] - theirs[i])


# ------------------------------------------------------------------- engine


def ref_payouts(contribs, folded, scores, button):
    """Plain-Python side-pot reference: layered pots, odd chip after button."""
    n = pj.NUM_SEATS
    payout, prev = [0] * n, 0
    for level in sorted(contribs):
        layer = level - prev
        prev = level
        if not layer:
            continue
        in_layer = [i for i in range(n) if contribs[i] >= level]
        eligible = [i for i in in_layer if not folded[i]]
        best = max(scores[i] for i in eligible)
        winners = sorted(
            (i for i in eligible if scores[i] == best),
            key=lambda i: (i - button - 1) % n,
        )
        pot = layer * len(in_layer)
        share, rem = divmod(pot, len(winners))
        for i in winners:
            payout[i] += share
        payout[winners[0]] += rem
    return payout


step = jax.jit(pj.step)
mask_fn = jax.jit(pj.legal_action_mask)


def play(state, actions):
    for a in actions:
        assert bool(mask_fn(state)[a]), (a, np.asarray(mask_fn(state)))
        state = step(state, a)
    return state


def test_blinds_and_fold_out():
    s = pj.init(jax.random.PRNGKey(0), 0)
    assert int(s.current) == 3 and int(s.max_bet) == 2
    assert s.total_contrib.tolist() == [0, 1, 2, 0, 0, 0]
    s = play(s, [0, 0, 0, 0, 0])  # everyone folds to the big blind
    assert bool(s.terminal)
    assert s.rewards.tolist() == [0, -1, 1, 0, 0, 0]


def test_check_down_and_showdown():
    s = pj.init(jax.random.PRNGKey(1), 0)
    s = play(s, [1] * 6)  # limp around, BB checks
    assert int(s.street) == 1 and int(s.current) == 1
    for _ in range(3):
        s = play(s, [1] * 6)  # check every street
    assert bool(s.terminal)
    r = np.asarray(s.rewards)
    assert r.sum() == 0
    assert r.tolist() == [
        p - c
        for p, c in zip(
            ref_payouts(s.total_contrib.tolist(), [False] * 6, s.scores.tolist(), 0),
            s.total_contrib.tolist(),
        )
    ]


def test_min_raise_and_incomplete_all_in_does_not_reopen():
    s = pj.init(jax.random.PRNGKey(2), 0)
    s = s._replace(stacks=s.stacks.at[4].set(9))
    s = play(s, [5])  # UTG pot-raises: 2 + max(2, 3+2) = 7
    assert int(s.street_contrib[3]) == 7 and int(s.min_raise_inc) == 5
    s = play(s, [8])  # seat 4 all-in for 9: incomplete raise (2 < 5)
    assert int(s.max_bet) == 9 and int(s.min_raise_inc) == 5
    s = play(s, [0, 0, 0, 0])  # everyone else folds
    assert int(s.current) == 3
    legal = np.asarray(mask_fn(s))
    assert legal[0] and legal[1] and not legal[2:].any()  # no re-raise allowed
    s = play(s, [1])  # call: heads-up all-in, runs out to showdown
    assert bool(s.terminal)
    contribs = s.total_contrib.tolist()
    assert contribs == [0, 1, 2, 9, 9, 0] and np.asarray(s.rewards).sum() == 0
    expected = ref_payouts(contribs, [True, True, True, False, False, True],
                           s.scores.tolist(), 0)
    assert s.rewards.tolist() == [p - c for p, c in zip(expected, contribs)]


def test_check_raise_is_reopened_by_full_raise():
    s = pj.init(jax.random.PRNGKey(3), 0)
    s = play(s, [1] * 6)  # limped pot, flop
    s = play(s, [1])  # SB checks
    s = play(s, [5])  # BB pot-bets 12
    s = play(s, [1, 0, 0, 0])  # seat 3 calls, rest fold; back to SB
    assert int(s.current) == 1
    assert np.asarray(mask_fn(s))[2:].any()  # SB may check-raise


def test_split_pot_odd_chip_goes_to_first_after_button():
    s = pj.init(jax.random.PRNGKey(4), 0)
    s = play(s, [1, 0, 0, 0, 0, 1])  # UTG limps, SB folds, BB checks: pot 5
    s = s._replace(scores=jnp.zeros(6, jnp.int32))  # force a tie at showdown
    for _ in range(3):
        s = play(s, [1, 1])  # check down (seat 2 first, then seat 3)
    assert bool(s.terminal)
    # seats 2 and 3 split 5 chips; odd chip to seat 2 (first after button)
    assert s.rewards.tolist() == [0, -1, 1, 0, 0, 0]


def test_full_game_fuzz_vs_reference():
    rng = np.random.default_rng(0)
    for hand in range(300):
        s = pj.init(jax.random.PRNGKey(hand), hand % 6)
        for _ in range(200):
            if bool(s.terminal):
                break
            legal = np.flatnonzero(np.asarray(mask_fn(s)))
            assert legal.size > 0
            s = step(s, int(rng.choice(legal)))
            assert (np.asarray(s.stacks) >= 0).all()
            assert (
                np.asarray(s.stacks) + np.asarray(s.total_contrib)
                == pj.STARTING_STACK
            ).all()
        assert bool(s.terminal)
        r = np.asarray(s.rewards)
        assert r.sum() == 0
        expected = ref_payouts(
            s.total_contrib.tolist(), [bool(f) for f in s.folded],
            s.scores.tolist(), int(s.button),
        )
        assert r.tolist() == [p - c for p, c in zip(expected, s.total_contrib.tolist())]


def test_batched_autoreset_conservation():
    batch = 64
    keys = jax.random.split(jax.random.PRNGKey(7), batch)
    states = jax.vmap(pj.init, in_axes=(0, None))(keys, 0)
    auto = jax.jit(jax.vmap(pj.step_autoreset))
    key = jax.random.PRNGKey(8)
    finished = 0
    for _ in range(400):
        key, k = jax.random.split(key)
        masks = jax.vmap(pj.legal_action_mask)(states)
        actions = jax.random.categorical(k, jnp.where(masks, 0.0, -1e9))
        states, rewards, terminal = auto(states, actions)
        assert (np.asarray(rewards).sum(axis=1) == 0).all()
        assert (np.asarray(states.stacks) >= 0).all()
        finished += int(np.asarray(terminal).sum())
    assert finished > batch  # every env cycles through many hands
