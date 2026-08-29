"""6-max No-Limit Texas Hold'em as a pure-JAX, fixed-shape state machine.

Game definition (the research-standard cash format):
- 6 players, 100bb starting stacks every hand (chips are integers, BB = 2).
- Blinds 1/2, no ante, no rake.
- Discrete action space: fold, check/call, six pot-fraction raises, all-in.
- Full no-limit rules: min-raise tracking, the incomplete-all-in-raise rule
  (an all-in raise smaller than a full raise does not reopen betting),
  side pots, split pots with odd chips going to the first winner after the
  button, uncalled bets returned via the side-pot layers.

Design notes:
- The whole deal (hole cards + board) is drawn at `init`; streets only reveal
  information, so `step` needs no randomness and is a pure function.
- Showdown scores for all six players are precomputed at `init` (the deal is
  fixed), so `step` never calls the hand evaluator.
- Everything is branch-free (jnp.where) so `init`/`step` jit and vmap cleanly;
  batched rollouts are `jax.vmap(step)` over a batch of states.

Rewards are net chip deltas for the hand, filled in on the terminal step
(zero-sum across seats; divide by BIG_BLIND for bb units).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from poker_jax.cards import hand_score

NUM_SEATS = 6
SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 200  # 100 big blinds

# Raise sizes as fractions of the pot after calling: (numerator, denominator).
RAISE_FRACTIONS = ((1, 3), (1, 2), (3, 4), (1, 1), (3, 2), (2, 1))
_FRAC_NUM = jnp.array([f[0] for f in RAISE_FRACTIONS], jnp.int32)
_FRAC_DEN = jnp.array([f[1] for f in RAISE_FRACTIONS], jnp.int32)

# Actions: 0 fold, 1 check/call, 2..7 pot-fraction raises, 8 all-in.
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_ALL_IN = 2 + len(RAISE_FRACTIONS)
NUM_ACTIONS = ACTION_ALL_IN + 1

_SEATS = jnp.arange(NUM_SEATS)


class State(NamedTuple):
    key: jax.Array  # PRNG key used for automatic resets
    hole: jax.Array  # (6, 2) int32 cards
    board: jax.Array  # (5,) int32 cards, revealed by street
    scores: jax.Array  # (6,) int32 showdown score of each player's final hand
    button: jax.Array  # int32 seat of the dealer button
    street: jax.Array  # int32: 0 preflop, 1 flop, 2 turn, 3 river
    current: jax.Array  # int32 seat to act (undefined when terminal)
    folded: jax.Array  # (6,) bool
    all_in: jax.Array  # (6,) bool
    acted: jax.Array  # (6,) bool: acted since the last full raise this street
    stacks: jax.Array  # (6,) int32 chips behind
    street_contrib: jax.Array  # (6,) int32 chips put in this street
    total_contrib: jax.Array  # (6,) int32 chips put in this hand
    max_bet: jax.Array  # int32 highest street_contrib
    min_raise_inc: jax.Array  # int32 minimum full-raise increment
    terminal: jax.Array  # bool
    rewards: jax.Array  # (6,) int32 net chips, set on the terminal step


def _first_seat_from(mask, start):
    """First seat s (clockwise from `start`, inclusive) with mask[s] set.

    Returns (seat, found). `seat` is arbitrary when nothing is set.
    """
    offset = (_SEATS - start) % NUM_SEATS
    candidate = jnp.where(mask, offset, NUM_SEATS + 1)
    best = candidate.min()
    return (start + best) % NUM_SEATS, best <= NUM_SEATS


def init(key, button):
    """Deal a fresh hand. `button` is the dealer seat (int32)."""
    key, deal_key = jax.random.split(key)
    deck = jax.random.permutation(deal_key, 52).astype(jnp.int32)
    hole = deck[: 2 * NUM_SEATS].reshape(NUM_SEATS, 2)
    board = deck[2 * NUM_SEATS : 2 * NUM_SEATS + 5]
    scores = hand_score(
        jnp.concatenate([hole, jnp.broadcast_to(board, (NUM_SEATS, 5))], axis=1)
    )

    button = jnp.asarray(button, jnp.int32) % NUM_SEATS
    sb_seat = (button + 1) % NUM_SEATS
    bb_seat = (button + 2) % NUM_SEATS
    blinds = (
        jnp.where(_SEATS == sb_seat, SMALL_BLIND, 0)
        + jnp.where(_SEATS == bb_seat, BIG_BLIND, 0)
    ).astype(jnp.int32)

    return State(
        key=key,
        hole=hole,
        board=board,
        scores=scores,
        button=button,
        street=jnp.asarray(0, jnp.int32),
        current=(button + 3) % NUM_SEATS,
        folded=jnp.zeros(NUM_SEATS, bool),
        all_in=jnp.zeros(NUM_SEATS, bool),
        acted=jnp.zeros(NUM_SEATS, bool),
        stacks=jnp.full(NUM_SEATS, STARTING_STACK, jnp.int32) - blinds,
        street_contrib=blinds,
        total_contrib=blinds,
        max_bet=jnp.asarray(BIG_BLIND, jnp.int32),
        min_raise_inc=jnp.asarray(BIG_BLIND, jnp.int32),
        terminal=jnp.asarray(False),
        rewards=jnp.zeros(NUM_SEATS, jnp.int32),
    )


def _raise_targets(state):
    """Street-contribution targets for every raise action of the current seat.

    Returns (targets, total_avail): targets is (len(RAISE_FRACTIONS) + 1,)
    covering the pot-fraction raises plus all-in, clamped to the player's
    chips. Raising "by X pot" means X times the pot after calling, on top of
    the call, and never less than the minimum full raise.
    """
    cur = state.current
    my_street = state.street_contrib[cur]
    to_call = state.max_bet - my_street
    pot_after_call = state.total_contrib.sum() + to_call
    raise_by = jnp.maximum(
        (_FRAC_NUM * pot_after_call) // _FRAC_DEN, state.min_raise_inc
    )
    total_avail = state.stacks[cur] + my_street
    frac_targets = jnp.minimum(state.max_bet + raise_by, total_avail)
    return jnp.concatenate([frac_targets, total_avail[None]]), total_avail


def legal_action_mask(state):
    """(NUM_ACTIONS,) bool mask of legal actions for the current seat."""
    cur = state.current
    to_call = state.max_bet - state.street_contrib[cur]
    active = ~state.folded
    others_can_respond = (
        (active & ~state.all_in & (_SEATS != cur)).astype(jnp.int32).sum() >= 1
    )
    # A seat may raise unless it already acted since the last full raise
    # (the incomplete-all-in rule) or no opponent could respond.
    raise_rights = (~state.acted[cur]) & others_can_respond
    targets, _ = _raise_targets(state)
    raise_legal = raise_rights & (targets > state.max_bet)
    mask = jnp.concatenate(
        [jnp.stack([to_call > 0, jnp.asarray(True)]), raise_legal]
    )
    return mask & ~state.terminal


def step(state, action):
    """Apply `action` for the current seat and advance the game."""
    cur = state.current
    action = jnp.asarray(action, jnp.int32)
    is_fold = action == ACTION_FOLD
    is_raise = action >= 2

    targets, total_avail = _raise_targets(state)
    raise_to = targets[jnp.clip(action - 2, 0, len(RAISE_FRACTIONS))]
    call_to = jnp.minimum(state.max_bet, total_avail)
    my_street = state.street_contrib[cur]
    target = jnp.where(is_raise, raise_to, jnp.where(is_fold, my_street, call_to))
    pay = target - my_street

    folded = state.folded.at[cur].set(state.folded[cur] | is_fold)
    stacks = state.stacks.at[cur].add(-pay)
    street_contrib = state.street_contrib.at[cur].set(target)
    total_contrib = state.total_contrib.at[cur].add(pay)
    all_in = state.all_in.at[cur].set((stacks[cur] == 0) & ~folded[cur])

    full_raise = is_raise & (target - state.max_bet >= state.min_raise_inc)
    min_raise_inc = jnp.where(
        full_raise, target - state.max_bet, state.min_raise_inc
    )
    max_bet = jnp.maximum(state.max_bet, target)
    acted = jnp.where(full_raise, _SEATS == cur, state.acted.at[cur].set(True))

    # --- resolve what happens next ---
    active = ~folded
    num_active = active.astype(jnp.int32).sum()
    fold_out = num_active == 1

    need_act = active & ~all_in & (~acted | (street_contrib < max_bet))
    street_over = ~need_act.any()
    next_actor, _ = _first_seat_from(need_act, cur + 1)

    can_still_bet = (active & ~all_in).astype(jnp.int32).sum() >= 2
    to_showdown = street_over & ((state.street == 3) | ~can_still_bet)
    to_next_street = street_over & ~to_showdown & ~fold_out

    pot = total_contrib.sum()
    winner_seat, _ = _first_seat_from(active, state.button + 1)
    fold_out_rewards = (
        jnp.where(_SEATS == winner_seat, pot, 0) - total_contrib
    ).astype(jnp.int32)
    showdown_rewards = (
        _showdown_payouts(total_contrib, folded, state.scores, state.button)
        - total_contrib
    )

    terminal = fold_out | (to_showdown & ~fold_out)
    rewards = jnp.where(
        fold_out,
        fold_out_rewards,
        jnp.where(to_showdown, showdown_rewards, jnp.zeros(NUM_SEATS, jnp.int32)),
    )

    street_first, _ = _first_seat_from(active & ~all_in, state.button + 1)
    return state._replace(
        street=state.street + to_next_street.astype(jnp.int32),
        current=jnp.where(to_next_street, street_first, next_actor),
        folded=folded,
        all_in=all_in,
        acted=jnp.where(to_next_street, jnp.zeros(NUM_SEATS, bool), acted),
        stacks=stacks,
        street_contrib=jnp.where(
            to_next_street, jnp.zeros(NUM_SEATS, jnp.int32), street_contrib
        ),
        total_contrib=total_contrib,
        max_bet=jnp.where(to_next_street, 0, max_bet),
        min_raise_inc=jnp.where(to_next_street, BIG_BLIND, min_raise_inc),
        terminal=terminal,
        rewards=rewards,
    )


def _showdown_payouts(total_contrib, folded, scores, button):
    """Distribute the pot at showdown, handling side pots and split pots.

    The pot is decomposed into layers between consecutive contribution levels;
    each layer is shared by the best unfolded hand(s) among the players who
    contributed to it. Odd chips go to the first winner after the button.
    Uncalled bets come back to their owner as a layer with one contributor.
    """
    eff_score = jnp.where(folded, -1, scores)
    levels = jnp.sort(total_contrib)
    payout = jnp.zeros(NUM_SEATS, jnp.int32)
    prev = jnp.asarray(0, jnp.int32)
    for k in range(NUM_SEATS):
        level = levels[k]
        layer = level - prev
        in_layer = total_contrib >= level
        layer_pot = layer * in_layer.astype(jnp.int32).sum()
        eligible = in_layer & ~folded
        best = jnp.where(eligible, eff_score, -1).max()
        winners = eligible & (eff_score == best)
        num_winners = winners.astype(jnp.int32).sum()
        share = layer_pot // jnp.maximum(num_winners, 1)
        remainder = layer_pot - share * num_winners
        first_winner, found = _first_seat_from(winners, button + 1)
        payout = payout + jnp.where(winners, share, 0)
        payout = payout.at[first_winner].add(jnp.where(found, remainder, 0))
        prev = level
    return payout


def step_autoreset(state, action):
    """`step`, but terminal hands immediately re-deal (button rotates).

    Returns (next_state, rewards, terminal) where rewards/terminal describe
    the hand that just ended (all zeros / False mid-hand). Vmap over a batch
    of states for training rollouts.
    """
    stepped = step(state, action)
    key, deal_key = jax.random.split(stepped.key)
    fresh = init(deal_key, (stepped.button + 1) % NUM_SEATS)._replace(key=key)
    next_state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(stepped.terminal, a, b), fresh, stepped
    )
    return next_state, stepped.rewards, stepped.terminal
