"""Vectorized 7-card poker hand evaluator in pure JAX.

Card encoding: an integer in [0, 52), card = rank * 4 + suit,
rank 0 = deuce ... 12 = ace, suit in [0, 4).

`hand_score` maps a (..., 7) array of cards to a (...,) int32 score where a
higher score is a strictly better hand. Scores are comparable across hands:
score = category << 20 | tiebreak, with 4-bit rank nibbles in the tiebreak.

Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush,
6 full house, 7 quads, 8 straight flush.

Everything is fixed-shape and branch-free so it jits and vmaps cleanly.
"""

import jax.numpy as jnp

NUM_RANKS = 13
NUM_SUITS = 4
# Ranks of the wheel straight A-2-3-4-5 as a bitmask (ace bit 12, 2..5 bits 0..3).
_WHEEL_MASK = 0b1000000001111


def _top_k_packed(mask, k):
    """Pack the k highest set ranks of a (..., 13) bool mask into nibbles.

    The highest rank ends up in the most significant nibble. If fewer than k
    bits are set the result is right-aligned; callers only compare within one
    hand category where the number of set bits is the same for every hand.
    """
    shape = mask.shape[:-1]
    acc = jnp.zeros(shape, jnp.int32)
    cnt = jnp.zeros(shape, jnp.int32)
    for r in range(NUM_RANKS - 1, -1, -1):
        take = mask[..., r] & (cnt < k)
        acc = jnp.where(take, acc * 16 + r, acc)
        cnt = cnt + take.astype(jnp.int32)
    return acc


def _top1(mask):
    return _top_k_packed(mask, 1)


def _rank_onehot(rank):
    """(...,) rank -> (..., 13) bool one-hot."""
    return jnp.arange(NUM_RANKS) == rank[..., None]


def _straight_top(bits):
    """Highest straight-top rank in a 13-bit rank mask, or -1 if none."""
    m = bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)
    top = jnp.full(bits.shape, -1, jnp.int32)
    for i in range(9):  # bit i set in m => straight with top rank i + 4
        top = jnp.where((m >> i) & 1 == 1, i + 4, top)
    wheel = (bits & _WHEEL_MASK) == _WHEEL_MASK
    return jnp.where((top < 0) & wheel, 3, top)


def hand_score(cards):
    """Score 7-card hands. cards: (..., 7) int32 in [0, 52). Returns (...,) int32."""
    ranks = cards // NUM_SUITS  # (..., 7)
    suits = cards % NUM_SUITS

    rank_counts = (_rank_onehot(ranks)).astype(jnp.int32).sum(axis=-2)  # (..., 13)
    suit_counts = (jnp.arange(NUM_SUITS) == suits[..., None]).astype(jnp.int32).sum(
        axis=-2
    )  # (..., 4)

    rank_mask = rank_counts > 0  # (..., 13)
    bits = (rank_mask.astype(jnp.int32) * (1 << jnp.arange(NUM_RANKS))).sum(axis=-1)

    has_flush = suit_counts.max(axis=-1) >= 5
    flush_suit = suit_counts.argmax(axis=-1)
    in_flush_suit = suits == flush_suit[..., None]  # (..., 7)
    flush_rank_mask = (
        (_rank_onehot(ranks) & in_flush_suit[..., None]).any(axis=-2)
    )  # (..., 13)
    flush_bits = (flush_rank_mask.astype(jnp.int32) * (1 << jnp.arange(NUM_RANKS))).sum(
        axis=-1
    )

    straight_top = _straight_top(bits)
    sflush_top = jnp.where(has_flush, _straight_top(flush_bits), -1)

    quad_mask = rank_counts == 4
    trip_mask = rank_counts == 3
    pair_mask = rank_counts == 2
    num_trips = trip_mask.astype(jnp.int32).sum(axis=-1)
    num_pairs = pair_mask.astype(jnp.int32).sum(axis=-1)

    is_sf = sflush_top >= 0
    is_quads = quad_mask.any(axis=-1)
    is_fh = (num_trips >= 2) | ((num_trips == 1) & (num_pairs >= 1))
    is_straight = straight_top >= 0
    is_trips = num_trips >= 1
    is_two_pair = num_pairs >= 2
    is_pair = num_pairs == 1

    # Tiebreaks per category (only the selected one matters).
    tb_sf = sflush_top
    q = _top1(quad_mask)
    tb_quads = q * 16 + _top1(rank_mask & ~_rank_onehot(q))
    t = _top1(trip_mask)
    fh_pair = _top1((rank_counts >= 2) & ~_rank_onehot(t))
    tb_fh = t * 16 + fh_pair
    tb_flush = _top_k_packed(flush_rank_mask, 5)
    tb_straight = straight_top
    tb_trips = t * 256 + _top_k_packed(rank_mask & ~_rank_onehot(t), 2)
    p1 = _top1(pair_mask)
    p2 = _top1(pair_mask & ~_rank_onehot(p1))
    tb_two_pair = (
        p1 * 256 + p2 * 16 + _top1(rank_mask & ~_rank_onehot(p1) & ~_rank_onehot(p2))
    )
    tb_pair = p1 * 4096 + _top_k_packed(rank_mask & ~_rank_onehot(p1), 3)
    tb_high = _top_k_packed(rank_mask, 5)

    cat = jnp.where(
        is_sf, 8, jnp.where(
            is_quads, 7, jnp.where(
                is_fh, 6, jnp.where(
                    has_flush, 5, jnp.where(
                        is_straight, 4, jnp.where(
                            is_trips, 3, jnp.where(
                                is_two_pair, 2, jnp.where(is_pair, 1, 0))))))))
    tb = jnp.where(
        is_sf, tb_sf, jnp.where(
            is_quads, tb_quads, jnp.where(
                is_fh, tb_fh, jnp.where(
                    has_flush, tb_flush, jnp.where(
                        is_straight, tb_straight, jnp.where(
                            is_trips, tb_trips, jnp.where(
                                is_two_pair, tb_two_pair,
                                jnp.where(is_pair, tb_pair, tb_high))))))))
    return (cat.astype(jnp.int32) << 20) | tb.astype(jnp.int32)
