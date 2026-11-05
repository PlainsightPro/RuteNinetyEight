"""Train WiezenLM, the hand evaluator used for bidding.

Input tokens: [CLS] + 13 own cards + turned trump card + seat (relative to the dealer).
Outputs (from CLS): trick distributions (0..13) for playing alone in each suit (declarer leads),
solo no-trump, "alleen" (turned trump, voorhand leads), team tricks with each other seat in the
turned trump suit, and the probability that a misery stays clean.

usage: python train/train_wiezenlm.py <out_dir> data1.bin [data2.bin ...]
"""
import argparse, base64, json, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REC = 25
HEADS = ["alone_c", "alone_d", "alone_h", "alone_s", "nt", "alleen", "pair1", "pair2", "pair3"]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)  # bidirectional: a hand is a set
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class WiezenLM(nn.Module):
    def __init__(self, d=64, layers=3, heads=4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads)
        self.card = nn.Embedding(52, d)
        self.suit = nn.Embedding(4, d)
        self.rank = nn.Embedding(13, d)
        self.kind = nn.Embedding(4, d)   # 0 cls, 1 own card, 2 turned card, 3 seat
        self.seat = nn.Embedding(4, d)
        self.cls = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 9 * 14 + 1)

    def forward(self, cards, turned, seat):
        B = cards.shape[0]
        def emb(c):
            return self.card(c) + self.suit(c // 13) + self.rank(c % 13)
        own = emb(cards) + self.kind.weight[1]
        tr = (emb(turned) + self.kind.weight[2]).unsqueeze(1)
        st = (self.seat(seat) + self.kind.weight[3]).unsqueeze(1)
        cl = (self.cls + self.kind.weight[0]).expand(B, 1, -1)
        x = torch.cat([cl, own, tr, st], 1)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, 0]))


def load(files):
    raw = np.concatenate([np.fromfile(f, dtype=np.uint8) for f in files]).reshape(-1, REC)
    return raw


def batch(raw, idx, augment):
    r = raw[idx]
    cards = r[:, :13].astype(np.int64)
    turned = r[:, 13].astype(np.int64)
    seat = r[:, 14].astype(np.int64)
    lab = r[:, 15:25].astype(np.int64)
    if augment:  # random suit permutation (relabels the alone_<suit> targets too)
        perm = np.array([np.random.permutation(4) for _ in range(len(idx))])
        rows = np.arange(len(idx))[:, None]
        cards = perm[rows, cards // 13] * 13 + cards % 13
        turned = perm[np.arange(len(idx)), turned // 13] * 13 + turned % 13
        alone = np.zeros((len(idx), 4), np.int64)
        alone[rows, perm] = lab[:, :4]
        lab = np.concatenate([alone, lab[:, 4:]], 1)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return t(cards), t(turned), t(seat), t(lab)


def losses(out, lab):
    tr = out[:, :126].view(-1, 9, 14)
    tricks = torch.cat([lab[:, :6], lab[:, 7:10]], 1)  # 9 trick heads
    ce = F.cross_entropy(tr.reshape(-1, 14), tricks.reshape(-1), reduction="none").view(-1, 9)
    bce = F.binary_cross_entropy_with_logits(out[:, 126], lab[:, 6].float(), reduction="none")
    return ce, bce, tr, tricks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count())
    raw = load(a.data)
    n = len(raw)
    perm = np.random.permutation(n)
    va, tr = perm[: min(10000, n // 10)], perm[min(10000, n // 10):]
    print(f"{n:,} samples ({len(tr):,} train / {len(va):,} val)")
    model = WiezenLM()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"WiezenLM: {nparam:,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)

    def evaluate():
        model.eval()
        with torch.no_grad():
            c, t, s, l = batch(raw, va, False)
            ce, bce, trl, tricks = losses(model(c, t, s), l)
            p = F.softmax(trl, -1)
            exp = (p * torch.arange(14)).sum(-1)
            mae = (exp - tricks.float()).abs().mean(0)
            # baseline: predict the training mean of each head
            base = (tricks.float() - tricks.float().mean(0)).abs().mean(0)
        model.train()
        return ce.mean(0), bce.mean().item(), mae, base

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        order = np.random.permutation(tr)
        for i in range(0, len(order) - a.bs + 1, a.bs):
            c, t, s, l = batch(raw, order[i:i + a.bs], True)
            ce, bce, _, _ = losses(model(c, t, s), l)
            loss = ce.mean() + bce.mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        ce, bce, mae, base = evaluate()
        print(f"epoch {ep}: CE {ce.mean():.3f}  misery BCE {bce:.3f}  expected-trick MAE {mae.mean():.2f} "
              f"(mean-predictor {base.mean():.2f})  {time.time() - t0:.0f}s", flush=True)

    metrics = {h: dict(ce=round(ce[i].item(), 3), mae=round(mae[i].item(), 3), baseline_mae=round(base[i].item(), 3))
               for i, h in enumerate(HEADS)}
    metrics["misery_bce"] = round(bce, 3)
    print(json.dumps(metrics, indent=1))
    os.makedirs(a.out, exist_ok=True)
    sd = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(sd, os.path.join(a.out, "wiezenlm.safetensors"))
    cfg = dict(model.cfg, params=nparam, heads_order=HEADS + ["misery_clean"], samples=n, metrics=metrics,
               inputs="[CLS] + 13 own cards + turned card + seat relative to dealer (0 dealer, 1 voorhand)")
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    model.eval()
    tests = []
    with torch.no_grad():
        c, t, s, l = batch(raw, va[:5], False)
        o = model(c, t, s)
        for i in range(5):
            tests.append(dict(cards=c[i].tolist(), turned=int(t[i]), seat=int(s[i]), out=o[i].tolist()))
    web = dict(config=cfg, tests=tests,
               weights={k: dict(shape=list(v.shape), b64=base64.b64encode(v.numpy().astype("<f2").tobytes()).decode())
                        for k, v in sd.items()})
    json.dump(web, open(os.path.join(a.out, "wiezenlm.web.json"), "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
