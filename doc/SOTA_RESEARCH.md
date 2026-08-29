# State of the Art in Poker AI — Research Notes (August 2026)

Research summary for deciding where to take neuron_poker next: what the current
SOTA in poker is, what the SOTA in neighboring games is, and which of those
techniques transfer best to this repo's gym-style, multi-player NLHE environment.

## TL;DR

- **Heads-up NLHE:** The strongest publicly known agent is **GTO Wizard AI**
  (formerly "Ruse"): deep counterfactual value networks + CFR-based continual
  resolving, trained purely by self-play. It beat Slumbot by **19.4 bb/100 over
  150k hands** and is now exposed as a public evaluation benchmark with an API.
- **Multiplayer (6-max):** **Pluribus** (2019) is still the landmark; nothing
  public has clearly surpassed it. The most interesting recent multiplayer
  result is **"Solly"** (Nov 2025), which reached elite-human level in
  *multi-player Liar's Poker* with a model-free actor-critic (DeepNash/R-NaD
  lineage) — no game-tree search at all.
- **Best transfer target for this repo:** an **AlphaHoldem-style end-to-end
  self-play RL agent** (PPO + self-play pool, no search). It beat Slumbot and
  DeepStack after ~3 days on a single PC and fits a gym environment naturally,
  unlike CFR-style solvers which need game-tree traversal. The principled
  alternative for 6-max is **R-NaD** (the DeepNash algorithm, implemented in
  OpenSpiel).
- **LLMs are not SOTA at poker.** On the 2026 GTO Wizard benchmark, frontier
  LLMs played zero-shot remain far below the solver baseline; fine-tuned
  poker LLMs (PokerBench, PokerGPT, SpinGPT) are interesting but weaker than
  dedicated RL/CFR agents per unit of compute.

## 1. The three algorithm families that matter

### 1.1 Search + learned value functions (the "absolute SOTA" line, heads-up)

Lineage: DeepStack (2017) → Libratus (2017) → **ReBeL** (FAIR, 2020) →
**Student of Games** (DeepMind, Science Advances 2023) → **LAMIR** (2025).

- ReBeL generalizes AlphaZero-style self-play + search to imperfect information
  by operating on *public belief states* and proved superhuman in HUNL.
- Student of Games unified perfect- and imperfect-information games (chess, Go,
  poker, Scotland Yard) with sound continual re-solving — the current academic
  reference algorithm.
- LAMIR ("Look-ahead Reasoning with a Learned Model in Imperfect Information
  Games", arXiv 2510.05048) is the newest step: it *learns* the abstracted
  model MuZero-style instead of requiring the game rules, and scales look-ahead
  to games where previous methods couldn't.
- **GTO Wizard AI** is essentially this family productionized (CFR + predictive
  neural networks, self-play trained, solves any 200bb spot in ~3s/street) and
  is the strongest publicly accessible HUNL agent today.

Cost: this is the strongest line but the heaviest engineering lift (belief-state
representation, depth-limited resolving at inference, counterfactual value nets),
and its guarantees are 2-player; multiplayer is heuristic.

### 1.2 End-to-end self-play deep RL, no search (best fit for this repo)

- **AlphaHoldem** (AAAI 2022): pseudo-siamese network over a card tensor + an
  action-history tensor, **Trinal-Clip PPO** loss, self-play against a pool of
  K-best historical checkpoints. Beat Slumbot and DeepStack, trained in
  **3 days on one machine**, decisions in ~3 ms on a single GPU.
- **AlphaExploitem** (arXiv 2605.09150, 2026) extends it with a hierarchical
  transformer over previous hands for **opponent modeling**, exploiting weak
  opponents while staying competitive vs. equilibrium players.
- Related 2025–26 work: "Beyond GTO: profit-maximizing agents" (arXiv
  2509.23747) — GTO baseline via MCCFR + an adaptive exploitation layer;
  RL-CFR (ICML 2024) — RL-learned *action abstraction* feeding CFR.

Caveat: naive self-play DQN/PPO does **not** converge to Nash in zero-sum
imperfect-info games (it cycles and stays exploitable). AlphaHoldem's pool-based
self-play mitigates this empirically; R-NaD (below) fixes it in a principled way.

### 1.3 Model-free equilibrium RL — the transferable "other game" SOTA

- **DeepNash / R-NaD** (DeepMind, Science 2022): mastered **Stratego** — a game
  with vastly more states than poker — with *model-free* actor-critic RL whose
  learning dynamics (reward transformation + follow-the-regularized-leader) are
  guaranteed to converge to Nash in 2-player zero-sum, instead of cycling.
  R-NaD is **implemented in OpenSpiel** and has been validated on Kuhn/Leduc.
- **Solly** ("Outbidding and Outbluffing Elite Humans: Mastering Liar's Poker",
  arXiv 2511.03724, Nov 2025): the same model-free actor-critic recipe reached
  **elite human level in multi-player Liar's Poker** — notable because it's a
  genuinely multiplayer imperfect-info result (unlike HUNL where most hands are
  effectively 2-player) and it outperformed reasoning LLMs. This is the
  strongest evidence that the R-NaD line transfers to multiplayer Hold'em.
- Same family: NFSP, Deep CFR / Single Deep CFR (PokerRL), DREAM, ESCHER, MMD
  (magnetic mirror descent). New in 2026: Parallel CFR (arXiv 2605.19928)
  reports ~7× lower exploitability than PokerRL at equal wall time.

### 1.4 LLM-based agents (interesting, not SOTA)

- **PokerBench** (AAAI 2025): 11k solver-graded decisions; all frontier LLMs
  underperform out of the box; fine-tuning helps but plateaus.
- **PokerGPT** (arXiv 2401.06781), **SpinGPT** (2025, ~78% solver agreement,
  claims +13.4 bb/100 vs Slumbot heads-up).
- **GTO Wizard Benchmark** (arXiv 2603.23660, 2026): public API + AIVAT
  variance reduction (statistical significance with ~10× fewer hands); frontier
  LLMs zero-shot remain well below the solver baseline.

## 2. What this means for neuron_poker

Current repo state: 6-player NLHE gym env + keras-rl DQN agent. Honest
assessment: keras-rl is unmaintained and DQN self-play is both practically weak
and theoretically wrong for this game class. Recommended path, in order:

1. **Modernize the plumbing.** Port the env to Gymnasium/PettingZoo API and
   replace keras-rl with a maintained stack (PyTorch + CleanRL-style PPO, or
   RLlib). This is a prerequisite for everything below.
2. **Build an AlphaHoldem-style agent** (main recommendation — best
   strength-per-effort, fits the gym design):
   - state = card tensor (hole/board per street) + action-history tensor,
   - PPO with Trinal-Clip loss,
   - self-play against a pool of K-best past checkpoints,
   - evaluate by head-to-head vs. the pool + the repo's equity agents.
3. **Add the principled fix for multiplayer:** swap the plain PPO objective for
   **R-NaD** (port from OpenSpiel) or at least NFSP-style anchoring, so 6-max
   training converges instead of cycling. The Solly result suggests this scales
   to elite level in multiplayer imperfect-info games.
4. **Add opponent modeling later** (AlphaExploitem-style transformer over hand
   histories) if the goal is beating weak/human opponents rather than
   approximating equilibrium.
5. **Evaluation:** validate the algorithm implementation on Leduc via OpenSpiel
   (exploitability is exactly computable there), then benchmark heads-up
   externally against **Slumbot** and/or the **GTO Wizard Benchmark API**, using
   AIVAT-style variance reduction.
6. **Only if aiming for absolute SOTA heads-up:** implement ReBeL/Student of
   Games-style depth-limited continual resolving with a counterfactual value
   network. Large effort; this is what GTO Wizard AI is.

## 3. Sources

- Pluribus: [Superhuman AI for multiplayer poker (Science)](https://www.science.org/doi/10.1126/science.aay2400)
- Libratus: [Superhuman AI for heads-up no-limit poker (Science)](https://www.science.org/doi/10.1126/science.aao1733)
- ReBeL: [Combining Deep RL and Search for Imperfect-Information Games (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf)
- Student of Games: [Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adg3256)
- LAMIR: [Look-ahead Reasoning with a Learned Model in Imperfect Information Games (arXiv 2510.05048)](https://arxiv.org/abs/2510.05048)
- AlphaHoldem: [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20394)
- AlphaExploitem: [arXiv 2605.09150](https://arxiv.org/abs/2605.09150)
- DeepNash/R-NaD: [Mastering Stratego (arXiv 2206.15378)](https://arxiv.org/abs/2206.15378), [DeepMind blog](https://deepmind.google/blog/mastering-stratego-the-classic-game-of-imperfect-information/)
- Solly / Liar's Poker: [arXiv 2511.03724](https://arxiv.org/abs/2511.03724)
- GTO Wizard AI: [GTO Wizard AI explained](https://blog.gtowizard.com/gto-wizard-ai-explained/), [benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/)
- GTO Wizard Benchmark: [arXiv 2603.23660](https://arxiv.org/abs/2603.23660)
- Beyond GTO (profit-maximizing agents): [arXiv 2509.23747](https://arxiv.org/abs/2509.23747)
- RL-CFR: [arXiv 2403.04344](https://arxiv.org/abs/2403.04344)
- PokerBench: [arXiv 2501.08328](https://arxiv.org/abs/2501.08328)
- PokerGPT: [arXiv 2401.06781](https://arxiv.org/abs/2401.06781)
- SpinGPT: [arXiv 2509.22387](https://arxiv.org/abs/2509.22387)
- Parallel CFR: [arXiv 2605.19928](https://arxiv.org/abs/2605.19928)
- Frameworks: [OpenSpiel](https://github.com/google-deepmind/open_spiel), [PokerRL](https://github.com/EricSteinberger/PokerRL), [RLCard](https://github.com/datamllab/rlcard), [awesome-poker-ai](https://github.com/PokerBotAI/awesome-poker-ai)
