# Training data

| folder | files | records | produced by |
|---|---|---|---|
| `v1/` | `bid_1..4.bin` | 192,000 × 25 bytes | `train/gen_bidding.js` (random shuffled deals, rollout labels) |
| `selfplay_round1/` | `sp_fast.bin`, `sp_bots_1..3.bin` | 60,000 + 6,880 × 184 bytes | `train/selfplay.js` (fast players / bots with WiezenLM v1) |
| `selfplay_round2/` | `sp2_bots_1..4.bin` | 18,000 × 184 bytes | `train/selfplay.js` (bots with WiezenLM v2) |

**25-byte records (v1):** 13 cards, turned card, seat relative to the dealer, then 10 labels:
- tricks alone in ♣ ♦ ♥ ♠
- tricks solo without trump
- tricks "alleen"
- misery clean (0/1)
- team tricks with the partner at +1, +2, +3

**184-byte records (self-play):** 13 cards, turned card, relative seat, then:
- 156 bytes of card-tracking belief (P × 255, card-major, seats +1..+3)
- the 10 labels above
- the real-outcome head (255 = none)
- the real outcome value
- the mode (0 = fast, 1 = bots)
- one spare byte

Card id = suit × 13 + rank (suits ♣ ♦ ♥ ♠, rank 0 = 2 … 12 = ace).

Retrain v3:

```bash
python3 train/train_wiezenlm2.py model --v1 data/v1/*.bin \
  --sp data/selfplay_round1/sp_fast.bin data/selfplay_round1/sp_bots_*.bin data/selfplay_round2/*.bin \
  --old model/v1 --prev model/v2
```
