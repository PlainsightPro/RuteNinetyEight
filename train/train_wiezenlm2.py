"""WiezenLM v2: hand evaluator that also sees the card-tracking belief, trained on bot-vs-bot self-play.

Tokens: [CLS] + 13 own cards + turned card + seat + 39 other cards, each other card carrying the
belief (P held by the player to my left / across / right) from card tracking.

Training signal:
  - counterfactual rollout labels for 10 contracts (every sample), as in v1
  - the REAL outcome of the contract the bots actually played (self-play "bots" samples), weight --real-weight
Data: v1 files (shuffled deals, 25-byte records; their belief is exact uniform) and train/selfplay.js
files (184-byte records).

usage: python train/train_wiezenlm2.py <out_dir> --v1 bid_*.bin --sp sp_*.bin [--old model_v1_dir]
"""
import argparse, base64, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_wiezenlm import Block, WiezenLM as WiezenLMv1  # noqa: E402

HEADS = ["alone_c", "alone_d", "alone_h", "alone_s", "nt", "alleen", "pair1", "pair2", "pair3"]


class WiezenLM2(nn.Module):
    def __init__(self, d=64, layers=3, heads=4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, version=2)
        self.card = nn.Embedding(52, d)
        self.suit = nn.Embedding(4, d)
        self.rank = nn.Embedding(13, d)
        self.kind = nn.Embedding(5, d)   # 0 cls, 1 own card, 2 turned card, 3 seat, 4 other card
        self.seat = nn.Embedding(4, d)
        self.loc = nn.Linear(3, d)        # belief about who holds an other card
        self.cls = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 9 * 14 + 1)

    def forward(self, cards, turned, seat, others, oloc):
        B = cards.shape[0]
        emb = lambda c: self.card(c) + self.suit(c // 13) + self.rank(c % 13)
        own = emb(cards) + self.kind.weight[1]
        tr = (emb(turned) + self.kind.weight[2]).unsqueeze(1)
        st = (self.seat(seat) + self.kind.weight[3]).unsqueeze(1)
        oth = emb(others) + self.kind.weight[4] + self.loc(oloc)
        cl = (self.cls + self.kind.weight[0]).expand(B, 1, -1)
        x = torch.cat([cl, own, tr, st, oth], 1)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, 0]))


def load_v1(files):
    r = np.concatenate([np.fromfile(f, dtype=np.uint8) for f in files]).reshape(-1, 25)
    n = len(r)
    cards, turned, rel = r[:, :13].astype(np.int64), r[:, 13].astype(np.int64), r[:, 14].astype(np.int64)
    loc = np.zeros((n, 52, 3), np.float16)
    # exact belief for a shuffled deck: the turned card is the dealer's, other cards spread over open slots
    for i in range(n):
        mine = set(cards[i].tolist())
        slots = np.array([13.0, 13.0, 13.0])  # seats +1, +2, +3
        k = (-rel[i]) % 4  # dealer relative to me
        if k != 0:
            slots[k - 1] -= 1
            loc[i, turned[i], k - 1] = 1
        p = slots / slots.sum()
        for c in range(52):
            if c not in mine and not (k != 0 and c == turned[i]):
                loc[i, c] = p
    lab = r[:, 15:25].astype(np.int64)
    return dict(cards=cards, turned=turned, rel=rel, loc=loc, lab=lab,
                rh=np.full(n, 255, np.int64), rv=np.zeros(n, np.int64), src=np.zeros(n, np.int64))


def load_sp(files):
    r = np.concatenate([np.fromfile(f, dtype=np.uint8) for f in files]).reshape(-1, 184)
    return dict(cards=r[:, :13].astype(np.int64), turned=r[:, 13].astype(np.int64), rel=r[:, 14].astype(np.int64),
                loc=(r[:, 15:171].reshape(-1, 52, 3) / 255.0).astype(np.float16), lab=r[:, 171:181].astype(np.int64),
                rh=r[:, 181].astype(np.int64), rv=r[:, 182].astype(np.int64), src=1 + r[:, 183].astype(np.int64))


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def others_of(cards):
    n = len(cards)
    mask = np.ones((n, 52), bool)
    mask[np.arange(n)[:, None], cards] = False
    return np.nonzero(mask)[1].reshape(n, 39)


def batch(D, idx, augment):
    cards, turned, seat = D["cards"][idx], D["turned"][idx], D["rel"][idx]
    oth = D["oth"][idx]
    oloc = D["loc"][idx][np.arange(len(idx))[:, None], oth].astype(np.float32)
    lab, rh, rv = D["lab"][idx].copy(), D["rh"][idx].copy(), D["rv"][idx]
    if augment:  # random suit permutation, consistently for cards, beliefs and suit-specific labels
        perm = np.array([np.random.permutation(4) for _ in range(len(idx))])
        rows = np.arange(len(idx))[:, None]
        pc = lambda c, rws: perm[rws, c // 13] * 13 + c % 13
        cards, oth = pc(cards, rows), pc(oth, rows)
        turned = pc(turned, np.arange(len(idx)))
        alone = np.zeros((len(idx), 4), np.int64)
        alone[rows, perm] = lab[:, :4]
        lab[:, :4] = alone
        m = rh < 4
        rh[m] = perm[np.arange(len(idx))[m], rh[m]]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return t(cards), t(turned), t(seat), t(oth), t(oloc), t(lab), t(rh), t(rv)


# real-outcome head index (selfplay.js) -> trick head of the network (or -1 = misery logit)
RH2HEAD = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8, 10: -1}


def losses(out, lab, rh, rv):
    tr = out[:, :126].view(-1, 9, 14)
    tricks = torch.cat([lab[:, :6], lab[:, 7:10]], 1)
    ce = F.cross_entropy(tr.reshape(-1, 14), tricks.reshape(-1), reduction="none").view(-1, 9).mean(1)
    bce = F.binary_cross_entropy_with_logits(out[:, 126], lab[:, 6].float(), reduction="none")
    real = torch.zeros(len(rh))
    has = torch.zeros(len(rh), dtype=torch.bool)
    for code, h in RH2HEAD.items():
        m = rh == code
        if not m.any():
            continue
        has |= m
        if h >= 0:
            real[m] = F.cross_entropy(tr[m, h], rv[m], reduction="none")
        else:
            real[m] = F.binary_cross_entropy_with_logits(out[m, 126], rv[m].float(), reduction="none")
    return ce, bce, real, has


def real_nll(out, rh, rv):
    """per-sample NLL of the actual outcome (only rows with a real label)"""
    _, _, real, has = losses(out, torch.zeros(len(rh), 10, dtype=torch.long), rh, rv)
    return real[has]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v1", nargs="*", default=[])
    ap.add_argument("--sp", nargs="+", required=True)
    ap.add_argument("--old", default=None, help="v1 model dir, for comparison")
    ap.add_argument("--prev", default=None, help="previous v2-architecture model dir (e.g. the last self-play round), for a paired comparison")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--real-weight", type=float, default=3.0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the training data to use (scaling runs)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(a.threads or os.cpu_count())

    sp = load_sp(a.sp)
    # hold out the last 15% of the bot-vs-bot samples (contiguous = whole hands/sessions)
    bots = np.nonzero(sp["src"] == 2)[0]
    hold = bots[int(len(bots) * 0.85):]
    keep = np.setdiff1d(np.arange(len(sp["src"])), hold)
    test = {k: v[hold] for k, v in sp.items()}
    parts = [{k: v[keep] for k, v in sp.items()}]
    if a.v1:
        parts.append(load_v1(a.v1))
    D = cat(parts)
    # extra held-out set (10k positions from the v1 / fast self-play data) for precise rollout-label metrics
    rng = np.random.RandomState(123)
    ho = rng.choice(len(D["cards"]), 10000, replace=False)
    ho = ho[D["src"][ho] != 2]
    test2 = {k: v[ho] for k, v in D.items()}
    rest = np.setdiff1d(np.arange(len(D["cards"])), ho)
    if a.frac < 1.0:
        rest = rng.choice(rest, int(len(rest) * a.frac), replace=False)
    D = {k: v[rest] for k, v in D.items()}
    D["oth"] = others_of(D["cards"])
    test["oth"] = others_of(test["cards"])
    test2["oth"] = others_of(test2["cards"])
    n = len(D["cards"])
    print(f"train: {n:,} samples ({(D['src'] == 0).sum():,} shuffled-deal v1, {(D['src'] == 1).sum():,} fast self-play, "
          f"{(D['src'] == 2).sum():,} bot-vs-bot, {(D['rh'] != 255).sum():,} real outcomes); "
          f"test: {len(test['cards']):,} bot-vs-bot samples, {(test['rh'] != 255).sum():,} real outcomes")

    model = WiezenLM2(d=a.d, layers=a.layers, heads=a.heads)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"WiezenLM v2: {nparam:,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    # oversample bot-vs-bot samples with a real outcome
    w = np.ones(n)
    w[D["rh"] != 255] = 4.0
    w /= w.sum()

    def evaluate(m, v2=True, T=None):
        T = test if T is None else T
        m.eval()
        ces, bces, rns = [], [], []
        with torch.no_grad():
            for i in range(0, len(T["cards"]), 2048):
                c, t, s, o, ol, lab, rh, rv = batch(T, np.arange(i, min(i + 2048, len(T["cards"]))), False)
                out = m(c, t, s, o, ol) if v2 else m(c, t, s)
                ce, bce, _, _ = losses(out, lab, rh, rv)
                ces.append(ce); bces.append(bce); rns.append(real_nll(out, rh, rv))
        m.train()
        ce, bce, rn = torch.cat(ces), torch.cat(bces), torch.cat(rns)
        return ce.mean().item(), bce.mean().item(), (rn.mean().item() if len(rn) else float("nan")), len(rn)

    t0 = time.time()
    for ep in range(a.epochs):
        for i in range(n // a.bs):
            idx = np.random.choice(n, a.bs, p=w)
            c, t, s, o, ol, lab, rh, rv = batch(D, idx, True)
            ce, bce, real, has = losses(model(c, t, s, o, ol), lab, rh, rv)
            loss = ce.mean() + bce.mean() + a.real_weight * (real.sum() / max(1, len(rh)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ce, bce, rn, k = evaluate(model)
        print(f"epoch {ep}: test rollout CE {ce:.3f}  misery BCE {bce:.3f}  real-outcome NLL {rn:.3f} (n={k})  {time.time() - t0:.0f}s", flush=True)

    metrics = dict(test_rollout_ce=round(ce, 4), test_misery_bce=round(bce, 4), test_real_nll=round(rn, 4), test_real_n=k)
    ce2, bce2, _, _ = evaluate(model, T=test2)
    trs = {kk: v[:10000] for kk, v in D.items()}
    ce3, bce3, _, _ = evaluate(model, T=trs)
    metrics.update(heldout10k_rollout_ce=round(ce2, 4), heldout10k_misery_bce=round(bce2, 4),
                   train10k_rollout_ce=round(ce3, 4), train10k_misery_bce=round(bce3, 4),
                   params=nparam, train_samples=int(n), d=a.d, layers=a.layers, frac=a.frac, tag=a.tag, seed=a.seed,
                   minutes=round((time.time() - t0) / 60, 1))
    print("SCALING " + json.dumps(metrics), flush=True)
    if a.no_save:
        return
    if a.old:
        from safetensors.torch import load_file
        old = WiezenLMv1()
        old.load_state_dict(load_file(os.path.join(a.old, "wiezenlm.safetensors")))
        oce, obce, orn, _ = evaluate(old, v2=False)
        metrics.update(v1_test_rollout_ce=round(oce, 4), v1_test_misery_bce=round(obce, 4), v1_test_real_nll=round(orn, 4))
        print(f"v1 on the same test set: rollout CE {oce:.3f}  misery BCE {obce:.3f}  real-outcome NLL {orn:.3f}")
    if a.prev:
        from safetensors.torch import load_file
        pc = json.load(open(os.path.join(a.prev, "config.json")))
        prev = WiezenLM2(d=pc["d"], layers=pc["layers"], heads=pc["heads"])
        prev.load_state_dict(load_file(os.path.join(a.prev, "wiezenlm.safetensors")))
        pce, pbce, prn, _ = evaluate(prev)
        pce2, _, _, _ = evaluate(prev, T=test2)
        # paired comparison on the real outcomes
        model.eval(); prev.eval()
        with torch.no_grad():
            c, t, s_, o, ol, lab, rh, rv = batch(test, np.arange(len(test["cards"])), False)
            dn = (real_nll(prev(c, t, s_, o, ol), rh, rv) - real_nll(model(c, t, s_, o, ol), rh, rv)).numpy()
        ci = 1.96 * dn.std() / np.sqrt(len(dn))
        metrics.update(prev_test_rollout_ce=round(pce, 4), prev_test_real_nll=round(prn, 4), prev_heldout10k_rollout_ce=round(pce2, 4),
                       real_nll_gain_vs_prev=round(float(dn.mean()), 4), real_nll_gain_vs_prev_ci95=round(float(ci), 4))
        print(f"previous model on the same test set: rollout CE {pce:.3f}  real-outcome NLL {prn:.3f}  "
              f"-> gain {dn.mean():.3f} +- {ci:.3f} (95%, paired)")
    print(json.dumps(metrics))

    os.makedirs(a.out, exist_ok=True)
    model.eval()
    sd = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(sd, os.path.join(a.out, "wiezenlm.safetensors"))
    cfg = dict(model.cfg, params=nparam, heads_order=HEADS + ["misery_clean"], samples=int(n), metrics=metrics,
               inputs="[CLS] + 13 own cards + turned card + seat + 39 other cards with belief (P left/across/right)")
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    tests = []
    with torch.no_grad():
        c, t, s, o, ol, *_ = batch(test, np.arange(3), False)
        outp = model(c, t, s, o, ol)
        for i in range(3):
            loc = np.zeros((52, 3), np.float32)
            loc[o[i].numpy()] = ol[i].numpy()
            tests.append(dict(cards=c[i].tolist(), turned=int(t[i]), seat=int(s[i]), loc=loc.reshape(-1).tolist(), out=outp[i].tolist()))
    web = dict(config=cfg, tests=tests,
               weights={k: dict(shape=list(v.shape), b64=base64.b64encode(v.numpy().astype("<f2").tobytes()).decode())
                        for k, v in sd.items()})
    json.dump(web, open(os.path.join(a.out, "wiezenlm.web.json"), "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
