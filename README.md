# Rute 98: Wiezen practice table with AI players (IWWA rules)

A complete Wiezen (Belgian whist) game for the browser: **you against three bots**, following the
IWWA *Specifiek Reglement Wiezen* (versie 4, 25-08-2026) and the IWWA *Puntentabel Wiezen/Whist*
(Federatie versie 1, 01-10-2025), the rules used at the
[Wereldkampioenschap Wiezen 2026](https://www.whistiwwa.com/tornooien/wereldkampioenschap-wiezen-2026)
(Turnhout, 29 November 2026). Links to the source documents are in `docs/SOURCES.md`.

Open `web/index.html` in a browser. It is a single 0.6 MB file and needs no server or network.

> **About the tournament.** This is a practice and analysis tool. Letting software choose your bids or
> cards during the real championship would be outside assistance. The IWWA regulations ask players to
> call the jury for irregularities such as "too much information" between players.
> Only use an AI at the event if IWWA explicitly allows it (info@iwwa.cards).

## How the bots play

```
            ┌──────────────── public information ─────────────────┐
            │ previous hand's tricks · who won them · deal 4-4-5   │
            │ or 4-5-4 · turned card · bids · cards played · voids │
            └───────────────────────────┬──────────────────────────┘
                                        ▼
  Belief (engine/belief.js)    exact posterior over (pile gathering order × cut position)
                               → samples of the hidden hands consistent with everything seen
                                        ▼
  Card play (engine/ai.js)     for every sampled deal × every candidate card:
                               fast heuristic play until each hand has ≤ 7 cards, then the
                               exact double-dummy solver (engine/dds.js); score the result
                               with the IWWA point table; play the best average
                                        ▼
  Bidding                      the same rollouts + the WiezenLM network prior
                               → trick distribution per contract → expected IWWA points per bid
```

### Card tracking

The regulations forbid shuffling after the first game (§1): the new dealer gathers the tricks, the
previous dealer cuts (at least 4 cards off, at least 4 left), and the dealer deals 4-4-5 or 4-5-4. All tricks were
public, so the new deck is a sequence of known 4-card blocks. The bot considers every order in which
the four trick piles could have been gathered (24) and every cut position (45). It computes the exact
likelihood of its own hand, the turned card and every card played so far, and samples deals from
that posterior.

**Learned habits.** People are not random when they gather and cut. After every hand all 52 cards
have been played, so the bots reconstruct how that hand's deck was gathered and cut, and learn per player:
- how deep they usually cut
- in which order the dealer picks up the piles (starting seat, clockwise or not)
- how they stack a trick they win: in play order, reversed, by seat (for example own card first, then
  clockwise; 24 possibilities), or sloppily

These become priors for the next deals. Each trick block is then a mixture of the tidy orders and the sloppy one.

One subtlety: starting the gathering one pile later and cutting that pile's size less gives exactly
the same deck. A finished hand therefore has several equally good explanations. The habits are fitted
jointly with EM over all finished hands, with random restarts. Consistent habits break the tie because
pile sizes differ from hand to hand.

Accuracy right after the deal, when the bot knows only its own 13 cards and the turned card
(simulated tables, 150 deals, `train/test_engine.js` reproduces a smaller version):

| | uniform prior | learned habits (after 12+ hands) |
|---|---|---|
| probability on the true owner of each hidden card | 0.80 | **0.89** |
| hidden cards located with >95% certainty | 53% | **65%** |
| plausible deck orders (effective number) | 5.4 | **2.0** |

Without tracking the probability on the true owner would be 0.33. The simulated players get random personal habits:
- cut depth: mean 14–38 cards, spread 2–6
- gathering: a fixed starting seat 85% of the time
- trick pickup: play order, a seat order or sloppy, with 60–95% consistency

These are assumptions about human behaviour, not measurements. The "Gewoontes aan tafel" panel shows
what the bots have learned, including your own cut depth. After an all-pass hand each player only knows
his own previous hand, and the bots model that case too.

### Bidding inference (optional)

Every bid says something about the bidder's hand: someone who passed rarely holds five trumps with honours.
With inference on, a bot replays the auction to recover what each player could have bid at their turn.
It then weights its sampled deals by how likely each player's actual bid was with the cards that deal gives them,
assuming they bid on expected points as the bots do. There is a 10% "anything goes" mixture, so players who
bid differently do not break it.

The likelihoods come from a small model (`model/bid/`, 30k parameters, about 1 ms per call). The full model is too slow
for the hundreds of calls this needs.

After the bidding, on deals where someone bid, the probability on the true owner of a hidden card
goes from 0.352 to 0.381 for cards of the players who bid (without card tracking).

### WiezenLM (`model/`, v3, two rounds of bot-vs-bot self-play)

A small transformer encoder (164k parameters) that evaluates a hand for bidding.

- **Input tokens:** `[CLS]`, your 13 cards, the turned card, your seat relative to the dealer, and one
  token for each of the 39 cards you don't hold. Each of those carries the card-tracking belief about
  who holds it (left, across, right).
- **Outputs:** a probability distribution over 0–13 tricks for each of these contracts:
  - playing alone with trump ♣, ♦, ♥ or ♠
  - solo without trump
  - "alleen" with the turned trump
  - a partnership with each of the other three seats
  - plus the probability that a misery stays clean

**How it was trained**

1. **v1 (`model/v1/`):** 192,000 hands from random shuffled deals. Each is labelled with what every contract would
   make when played out on the true deal: heuristic early tricks, then the last 7 tricks solved exactly.
2. **Self-play (`train/selfplay.js`):** the full bots played 1,720 hands against each other in sessions
   without shuffling, with card tracking, learned habits, real bidding and real play. For every seat and
   deal it records the hand, the tracking belief at the start, the counterfactual labels, and the
   **actual result of the contract the bots really played**: 1,460 real outcomes. Another 15,000
   hands with quick rule-based players add more tracking situations, with counterfactual labels only.
3. **v2 (`train/train_wiezenlm2.py`):** trained on all of it (258,000 samples).
   - Loss: cross-entropy on the counterfactual labels, plus the real bot-vs-bot outcomes with weight 3,
     oversampled ×4.
   - Augmentation: suit permutation.
   - 6 epochs, about 18 minutes on a CPU.

4. **Round 2 → v3:** the bots, now using v2, played 4,500 more hands (2,500 real outcomes). v3 was
   trained on everything: 264k samples, including 3,460 real outcomes (the rest is held out).
   v1 and v2 stay in `model/v1/` and `model/v2/`.

**Result on held-out round-2 hands** (555 real contract outcomes, never seen by v2 or v3; NLL, lower is better):

| contract | n | v1 | v2 | v3 |
|---|---|---|---|---|
| all | 555 | 2.024 | 1.860 | **1.687** |
| vraag / meegaan | 324 | 2.522 | 2.271 | **2.007** |
| miserie | 70 | 0.635 | 0.567 | **0.555** |
| alleen / abondance / solo | 161 | 1.626 | 1.595 | **1.536** |

v3 vs v2: 0.17 ± 0.03 better (95%, paired). v3 vs v1: 0.34 ± 0.05.

Two caveats:
- **Part of the gain is distribution.** v3 has seen games of v2-bots, the same kind of games as the
  test set, and v2 had not. That is the point of iterating, but it means the round-over-round gain is partly that.
- **Counterfactual labels get slightly worse.** On the "all cards visible" labels v3 scores 1.408 against
  1.392 for v2, because training puts more weight on real results.

The bots became more cautious along the way: real outcomes per hand dropped from 0.85 (round 1) to 0.57
(round 2), because v2 expects lower real results than the perfect-information labels suggest.
Whether more caution also wins more points needs a head-to-head match.

**Round 1 result** (v1 vs v2, 241 real outcomes from round 1):

| | v1 | v2 |
|---|---|---|
| NLL of the actual outcome (lower is better) | 2.161 | **1.947** |
| vraag / meegaan (188) | 2.464 | 2.233 |
| miserie (33) | 0.542 | 0.476 |
| alleen / abondance / solo (20) | 1.977 | 1.686 |

The paired difference is 0.21 ± 0.06 nats per outcome (95%), so v2 puts about 24% more probability on
what really happened.

The gain comes from **learning from real games**. Perfect-information labels are systematically off from what bots
achieve with hidden cards, and v2 corrects for that. The tracking-belief input helps only a
little so far: blanking it costs 0.015 ± 0.005 nats. On counterfactual labels v1 and v2 are equal (1.425).
v1 MAE figures on shuffled deals, for reference:
- alone in a suit: 0.49 tricks (baseline 1.27)
- solo without trump: 0.54 (1.22)
- alleen: 0.58 (1.35)
- partnership: 1.70 (2.06)

During bidding the network is blended with simulations on sampled deals.

Files:
- `wiezenlm.safetensors`: weights
- `config.json`: architecture and metrics
- `wiezenlm.web.json`: float16 browser bundle

`engine/wiezenlm.js` runs v1 and v2 without dependencies and matches PyTorch to within 0.005 logits.

### Scaling experiment (`model/scaling.jsonl`)

Same v2 architecture and training recipe, on the same 248k training samples with an equal number of
training steps. Lower is better.

| model | parameters | data | held-out CE (10k) | bot-vs-bot CE (1,032) | real-outcome NLL (241) |
|---|---|---|---|---|---|
| S (d=32, 2 layers) | 32k | 100% | 1.501 | 1.587 | 2.143 |
| M (d=64, 3 layers), 2 seeds | 164k | 100% | 1.394 / 1.445 | 1.470 / 1.548 | 1.969 / 1.926 |
| M, 2 seeds | 164k | 50% | 1.447 / 1.390 | 1.531 / 1.482 | 2.063 / 1.841 |
| M | 164k | 25% | 1.447 | 1.532 | 2.091 |
| L (d=128, 4 layers) | 820k | 100% | **1.358** | **1.442** | **1.915** |

What this shows:
- **Run-to-run noise is about ±0.05.** The same M model on the same data differs by 0.05 between two seeds.
- **32k → 164k parameters clearly helps,** by about 0.08, beyond the noise.
- **164k → 820k gives perhaps a small gain** (about 0.06 on the held-out set, from one seed). That is barely above the noise,
  for 2.7× the training time.
- **More of the same data does not help measurably.** 50% and 100% of the data give the same result (averages 1.42 and 1.42).
  The counterfactual labels are saturated at this size.
- **Conclusion:** returns flatten well before a million parameters, so a 3B model would not help.
  New kinds of signal, such as more real bot-vs-bot outcomes or more iterations of self-play, are the way forward.

## Results

Two player types share each table: seats 0 and 2 are one type, seats 1 and 3 the other, swapped every hand.
Contracts form partnerships dynamically, so the measure is the **difference in average IWWA points per
player per hand** between the two types, with 95% confidence intervals (`train/match.js`).

| matchup | hands | difference per player per hand |
|---|---|---|
| bot vs. rule-based club player | 840 | **+2.5 ± 0.6** |
| bot with card tracking vs. same bot without | 310 | +0.9 ± 1.1 (not yet significant) |
| bots with WiezenLM v3 vs. same bots with v1 | 2,890 | +0.00 ± 0.38 (no measurable difference) |

**Duplicate format** (`train/duplicate.js`): every deal is played twice with the same cards and the two bot
types swapping seats, so card luck largely cancels out. Decks are shuffled every deal, so card tracking plays no role.

| matchup | deals (×2 hands) | difference per player per hand |
|---|---|---|
| v3 vs v1 | 700 | +0.05 ± 0.25 |
| v3 + bidding inference vs v3 | 990 | −0.05 ± 0.22 |
| v3 + deeper search vs v3 | 1,080 | +0.14 ± 0.21 |

Deeper search here means exact double-dummy from 8 cards instead of 7, and 30 sampled deals instead of 20.

- **No difference is statistically significant.** All three effects are smaller than about a quarter point per player per hand.
- **Deeper search leans positive** but costs 2–3× the compute. In the browser the bots work within a time budget,
  so deeper search would mean fewer sampled deals, and the default stays at 7 cards.
- **Bidding inference** (below) makes the bots' picture of the hidden hands measurably sharper, but against bots
  it does not translate into points. It is available as an experimental setting ("Bots lezen de biedingen").

The rule-based player bids on trump length and honours and plays sound heuristics (second hand low,
third hand high, cash winners, trump when the opponents win). The bots win mainly by bidding:
they bid only contracts with positive expected points under the IWWA table, and the table punishes failed
contracts hard (a made vraag scores +2, one trick short scores −6).

**v3 vs v1:** v3 predicts real outcomes much better, yet that does not show up in points (yet).
- **The network only affects bidding.** Card play does not use it.
- **Most bidding decisions are clear-cut.** On the same 94 opening-bid situations with identical
  simulations, v1 and v3 bots chose differently only 10% of the time (15% if the bot relied on the network alone).
- **The difference per hand is small.** The resulting effect is probably a few tenths of a point per
  player per hand, below what 2,890 hands can detect (±0.38). A duplicate format, where the same deals
  are played with the networks swapped, would cancel the card luck and is the right next test.

The tracking advantage is not measurable yet: two full bots at the same table differ by less than the noise
after 310 hands. Tracking is what lets a strong human or bot read the unshuffled deck. The belief is exact
and accurate (78–80% on the true owner of each hidden card), but turning that into points
evidently needs more than PIMC on its own. More hands are needed to settle this.

## Rules implemented

- **Contracts, low to high:**
  1. vragen en meegaan (8 tricks)
  2. abondance 9 and 10
  3. miserie
  4. abondance 11 and 12
  5. troel / troela
  6. miserie op tafel
  7. solo (own trump or no trump)
  8. solo slim (turned trump)
- **Bidding rules:**
  - If nobody joins a "vraag", the asker may pass or go **alone** (5 tricks).
  - Abondance in the turned suit beats an equal-level abondance in another suit.
  - Several players may play (open) miserie at the same time.
- **Troel:** must be announced. With 3 aces, the partner holds the 4th ace and leads it; that ace's suit is trump.
  With 4 aces (troela), ♥ is trump and the partner holds ♥H (or the next highest heart) and leads it.
  The other players can still overcall with miserie op tafel, solo or solo slim.
- **Leads:** abondance and solo are led by the declarer. For miserie op tafel the hand is laid open before the first card.
- **Scoring:** the IWWA point table exactly. That includes the 13-trick doublings (30→60, 7→14, 14→28, 12→24) and
  the miserie grid with base 21 (op tafel: 42). `train/test_engine.js` checks every case against the table.

### Assumptions (where the regulations are silent)

- **0 tricks:** the table starts at 1 trick, so the 0-trick values are extrapolated.
  Vraag −20, alleen −36, troel −20, troela −22.
- **Miserie:** played without trump, and the player left of the dealer leads.
- **Four miserie players** (not in the grid): +7·(4−s) per clean player and −7·s per failed player.
- **All pass:** no points. The deal passes to the next player and the hands are gathered without shuffling.
- **Gathering for tracking:** the new dealer gathers the piles in some order, the order within a trick is random,
  and the previous dealer cuts at any allowed position. The bots' tracking uses exactly this model.
- **No claims:** all 13 tricks are always played.
- **Not implemented:** the dealer exchanging the turned card when playing miserie, and the jury check on high contracts.

## Repository layout

```
engine/dds.js        double-dummy solver (alpha-beta, rank-compressed transposition table,
                     equivalent cards, quick tricks, MTD(f)); verified against brute force
engine/rules.js      rules, bidding, dealing, collecting, IWWA scoring
engine/belief.js     card tracking: posterior over deck orders, world sampling
engine/ai.js         heuristic policy, rollouts, PIMC card play, expected-value bidding
engine/wiezenlm.js   network inference (browser + Node)
train/gen_bidding.js training data for WiezenLM v1
train/selfplay.js    bot-vs-bot self-play data (real outcomes + tracking beliefs) for v2
train/train_wiezenlm.py   v1
train/train_wiezenlm2.py  v2 (self-play)
train/match.js       strength evaluation (alternating seats)
train/duplicate.js   duplicate-format matches (same deals, seats swapped)
train/test_engine.js engine tests
web/                 template, UI, build script → web/index.html
docs/SOURCES.md      links to the IWWA regulations and point table
data/                training data (v1 rollout labels, two rounds of self-play), see data/README.md
```

## Rebuild

```bash
node train/test_engine.js
for i in 1 2 3 4; do node train/gen_bidding.js 12000 $((100+i)) build/bid_$i.bin & done; wait   # ~15 min
python3 train/train_wiezenlm.py model/v1 build/bid_*.bin                                      # v1, ~8 min, needs torch + safetensors
for i in 1 2 3; do node train/selfplay.js bots 900 $((200+i)) build/sp_bots_$i.bin model/v1/wiezenlm.web.json & done
node train/selfplay.js fast 15000 299 build/sp_fast.bin; wait                                  # bot-vs-bot self-play, ~1 h
python3 train/train_wiezenlm2.py model/v2 --v1 build/bid_*.bin --sp build/sp_*.bin --old model/v1   # v2, ~18 min
for i in 1 2 3 4; do node train/selfplay.js bots 1500 $((300+i)) build/sp2_bots_$i.bin model/v2/wiezenlm.web.json & done; wait
python3 train/train_wiezenlm2.py model --v1 build/bid_*.bin --sp build/sp_fast.bin build/sp_bots_*.bin build/sp2_bots_*.bin \
  --old model/v1 --prev model/v2                                                              # v3 (round 2), ~17 min
python3 web/build.py model/wiezenlm.web.json web/index.html
node train/match.js 400 1 bot-vs-rules model/wiezenlm.web.json
```

## Limitations

- **PIMC samples deals and then plays as if every card were visible.** It does not plan to hide
  information or deliberately signal to a partner.
- **The first tricks use a heuristic.** Each simulation plays them with a fast rule-based policy and
  solves only the last seven tricks exactly, so long-range planning in the opening is approximate.
- **Bidding inferences are limited.** Other players' bids are used only through the contract
  (who asked or joined), not as soft evidence about their hands.
- **Real tables may differ from the habit model.** The tracking adapts to habits of the kinds above
  (cut depth, gathering order, trick pickup). Other habits, such as squaring piles or partial riffles,
  are not modelled, so they would show up as "sloppy" and make tracking less certain.
