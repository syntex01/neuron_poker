# Beating SOTA in 6-max No-Limit Hold'em — Attack Plan (August 2026)

Goal: clearly beat the state of the art in the most common form of Texas
Hold'em — **6-max no-limit cash game** — against both bots and real players,
with an evaluation of the same standard as the Pluribus Science paper.

## 1. What the bar actually is

- The reigning result is **Pluribus** (Brown & Sandholm, Science 2019). Its
  numbers: won at ~**5 bb/100 (AIVAT-adjusted, p ≈ 0.02)** over **10,000 hands
  in the 5-humans + 1-AI format** against pros, plus 5,000 hands per human in
  the 1H+5AI format vs. Ferguson and Elias.
- **Pluribus is not playable or reproducible**: no code, no blueprint release,
  only the paper + supplement; public reimplementations are incomplete. So
  "beating SOTA" cannot mean sitting down against Pluribus. It has to be
  established the way the original paper did:
  1. beat the strongest available bots (and a faithful Pluribus-style baseline
     you build yourself), and
  2. beat strong humans by a **larger, statistically cleaner margin** than
     5 bb/100 under AIVAT.
- Game definition to standardize on: 6-max NLHE, 100bb starting stacks each
  hand (Pluribus used fixed 10,000-chip stacks / 50–100 blinds), no rake, no
  straddles/antes. That is both the research standard and the most common
  online cash format.

## 2. Why this is winnable now (the three gaps in Pluribus)

1. **Pluribus was compute-tiny and abstraction-bound.** Its blueprint took
   ~12,400 CPU core-hours (≈ $150 of cloud) of MCCFR over hand-crafted card
   and action abstractions; real-time search used just 4 precomputed
   continuation strategies at the depth limit. Every part of that pipeline is
   2019 technology. Modern neural, abstraction-free methods trained on
   billions of hands (GPU-vectorized simulators, transformer policies) have
   since beaten search-based agents heads-up (AlphaHoldem vs. DeepStack/
   Slumbot) at trivial cost. Nobody has published the "scale it up 1000x"
   version for 6-max — that's the open lane.
2. **Multiplayer has no equilibrium safety net anyway — and model-free RL just
   proved itself there.** Pluribus's theoretical guarantees don't extend past
   2 players; its strength was empirical. The Solly result (Liar's Poker,
   Nov 2025) showed a model-free actor-critic (DeepNash/R-NaD lineage) reaching
   elite human level in a genuinely *multiplayer* imperfect-info game,
   outperforming reasoning LLMs. Scaling that recipe to 6-max NLHE is the most
   direct shot at surpassing Pluribus's core.
3. **Pluribus was a fixed strategy.** It never adapted to opponents. Against
   real players, the winnings ceiling of a static quasi-equilibrium is far
   below that of an adaptive agent. An opponent-modeling layer
   (AlphaExploitem-style) is how you don't just beat the pros, but beat them
   by a margin that makes the claim unambiguous.

## 3. Proposed system ("beat Pluribus" architecture)

### Phase 0 — Simulation at scale (the real moat)
- GPU-native, fully vectorized 6-max NLHE engine in JAX (start from **Pgx**'s
  poker environments or write a custom one), with a batched GPU hand
  evaluator. Target: 10⁷–10⁸ hands/sec on one node. AlphaHoldem needed ~2.7B
  hands for heads-up; expect ≥10× that for 6-max.
- Everything downstream depends on this; it is also what almost no academic
  group has bothered to build for 6-max.

### Phase 1 — Equilibrium core (blueprint-free)
- Transformer policy/value net over the *full public action history* plus
  private cards — no card abstraction, no information abstraction. Bet sizing
  via a fine discrete grid (e.g. fractions of pot + all-in) or a parameterized
  raise-size head; RL-CFR (ICML 2024) shows learned action abstraction beats
  hand-crafted.
- Training objective: **R-NaD / MMD-style regularized self-play** (converges
  instead of cycling; validated at scale by DeepNash and Solly), combined with
  a **league of past checkpoints + diverse exploiter agents** (AlphaHoldem's
  K-best pool, PSRO-flavored) to police exploitability empirically in the
  multiplayer setting.
- Milestone gates: (a) exact-exploitability sanity checks on Leduc via
  OpenSpiel; (b) local best response (LBR) lower bounds on full 6-max; (c)
  crush all public bots/agents.

### Phase 2 — Test-time search on top
- Depth-limited continual re-solving with a learned multiplayer value network:
  Pluribus's continuation-strategy trick at the depth limit, but neural values
  instead of abstraction buckets (ReBeL/Student-of-Games machinery, applied
  heuristically to 6 players; LAMIR-style learned abstractions if subgames are
  too big).
- Rationale: AlphaHoldem showed search isn't strictly necessary heads-up, but
  most of Pluribus's strength came from search. Core-without-search must beat
  the Pluribus-style baseline; core-with-search is the margin of safety for
  "clearly beat".

### Phase 3 — Safe exploitation layer (for humans and weaker bots)
- Opponent-embedding transformer over the hands observed so far in a session
  (AlphaExploitem), trained against a broad opponent population: weakened/
  biased checkpoints, rule-based players, and behavior-cloned policies from
  large human hand-history datasets.
- Constrain deviation from the equilibrium core (restricted Nash response /
  bounded-EV-loss deviations) so the agent exploits without becoming
  exploitable. This is the piece Pluribus explicitly did not have, and the
  main lever for a headline-sized win rate vs. real players.

## 4. Evaluation plan (how the claim sticks)

1. **Build the counterfactual Pluribus.** Since the original is unavailable,
   implement a faithful modern Pluribus-style baseline (MCCFR blueprint +
   depth-limited search, per the paper/supplement) and beat *it* decisively in
   the bot league. This is required for any credible "surpasses Pluribus"
   claim.
2. **Bot league with AIVAT**: round-robin vs. the Pluribus-style baseline, the
   Phase-1/2 ablations, open agents; heads-up side-checks vs. **Slumbot** and
   the **GTO Wizard Benchmark API** (AIVAT-style variance reduction, ~10×
   fewer hands for significance).
3. **Human study replicating the Science protocol**: 10,000+ hands, 5H+1AI
   with rotating professional players, real monetary incentives, AIVAT
   scoring, preregistered analysis. Target a win rate meaningfully above
   Pluribus's ~5 bb/100 with a tighter confidence interval.
4. All human/bot play in a sanctioned research setting (own platform or
   research partner). Not on commercial real-money sites — that violates their
   terms and gambling law in many jurisdictions, and would poison the result's
   credibility anyway.

## 5. Resource reality check

- Pluribus: ~$150 CPU compute. AlphaHoldem: 1 PC, 3 days, 8 GPUs→1 GPU scale.
  Solly: unpublished but lab-scale. A serious Phase 0–2 run is on the order of
  **single-digit GPU-node-weeks per major experiment** — small-lab feasible;
  the scarce resources are engineering time (simulator, distributed RL loop)
  and the human evaluation (player recruitment + incentives, realistically
  tens of thousands of dollars).
- Biggest technical risks: R-NaD/league stability in 6-player at scale (no
  theory, only Solly-style evidence), bet-size discretization hurting vs.
  humans who probe odd sizes, and multiplayer value nets for search being
  under-explored territory. Each has a fallback (league-only training,
  finer sizing grids + search-time refinement, no-search Phase 1 agent).

## 6. Suggested first three concrete steps

1. Prototype the JAX 6-max engine (or benchmark Pgx's poker env) and measure
   hands/sec on one GPU.
2. Reproduce AlphaHoldem heads-up on that engine (known-good recipe, validates
   the whole stack against Slumbot).
3. Swap in the R-NaD objective + league and go to 6 players; stand up the
   Pluribus-style MCCFR baseline in parallel as the sparring target.
