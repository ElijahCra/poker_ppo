#!/usr/bin/env python3
"""
policy_audit.py — measure WHERE the policy puts fold/call/raise mass,
bucketed by hand strength, and what deployment-time sharpening would do.

Drives the C++ `--play` REPL through N self-played hands (both seats
sampled from the policy, matching the training distribution), logging the
FULL action distribution at every decision plus a hand-strength bucket:
  - preflop: Chen-formula tiers (premium/strong/medium/weak/trash)
  - postflop: made-hand category + draw flag from the obs hand features

Because full prob vectors are logged, sampling-layer fixes (temperature,
min-prob filtering à la DeepNash) are evaluated OFFLINE from the same data
— no re-runs needed. The diagnostic question: is weak-hand raise mass
thinly spread across raise sizes (quantal-response noise from the entropy
bonus → fix at the sampling layer) or concentrated (learned belief → fix
in training)?

Usage (inside WSL, repo root):
    python3 tools/policy_audit.py --model poker_ppo_model_nlhe_full_52.pt \
        --hands 1500 --out /tmp/audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play import Engine  # noqa: E402  (reuse the REPL wrapper)

ROUND_NAMES = ["preflop", "flop", "turn", "river"]

# Obs layout constants (ObservationLayout::build with both blocks on).
CARD_SLOTS = 52
STATIC_OFF = 2 * CARD_SLOTS          # 104
FEAT_BASIC = 10
HAND_FEATS_OFF = STATIC_OFF + FEAT_BASIC
HAND_FEATS_PER_ROUND = 5
HAND_CATEGORY_COUNT = 9


# ─── Hand-strength buckets ────────────────────────────────────────────────


def chen_score(c1: int, c2: int) -> float:
    """Chen formula. card_id = (rank << 2) | suit, rank 0..12 = 2..A."""
    r1, s1 = (c1 >> 2) & 0xF, c1 & 3
    r2, s2 = (c2 >> 2) & 0xF, c2 & 3
    hi, lo = max(r1, r2), min(r1, r2)
    pts = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2.0)
    if r1 == r2:
        return max(5.0, pts * 2)
    score = pts
    if s1 == s2:
        score += 2
    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:  # both below Q, connected-ish
        score += 1
    return score


def preflop_bucket(c1: int, c2: int) -> str:
    s = chen_score(c1, c2)
    if s >= 10: return "premium"
    if s >= 8:  return "strong"
    if s >= 6:  return "medium"
    if s >= 4:  return "weak"
    return "trash"


def postflop_bucket(obs: list[float], rnd: int) -> str:
    base = HAND_FEATS_OFF + rnd * HAND_FEATS_PER_ROUND
    cat = round(obs[base + 0] * HAND_CATEGORY_COUNT)
    flush_draw = obs[base + 1] > 0.5
    str_outs = obs[base + 2] * 8.0
    has_draw = flush_draw or str_outs >= 4
    if cat >= 2:  return "twopair+"
    if cat == 1:  return "pair"
    return "air+draw" if has_draw else "air"


PRE_BUCKETS = ["trash", "weak", "medium", "strong", "premium"]
POST_BUCKETS = ["air", "air+draw", "pair", "twopair+"]


# ─── Offline sharpening variants ─────────────────────────────────────────


def sharpen(probs: list[float], temp: float = 1.0, min_p: float = 0.0) -> list[float]:
    """Temperature + DeepNash-style min-prob filtering, from probs alone."""
    p = [x ** (1.0 / temp) if x > 0 else 0.0 for x in probs]
    z = sum(p) or 1.0
    p = [x / z for x in p]
    if min_p > 0:
        p = [x if x >= min_p else 0.0 for x in p]
        z = sum(p)
        if z <= 0:  # everything filtered → keep argmax
            m = max(range(len(probs)), key=lambda i: probs[i])
            p = [0.0] * len(probs)
            p[m] = 1.0
        else:
            p = [x / z for x in p]
    return p


VARIANTS = {
    "raw":      dict(temp=1.0, min_p=0.0),
    "t0.5":     dict(temp=0.5, min_p=0.0),
    "min0.10":  dict(temp=1.0, min_p=0.10),
    "t.5+m.10": dict(temp=0.5, min_p=0.10),
}


# ─── Aggregation ──────────────────────────────────────────────────────────


class Agg:
    def __init__(self):
        self.n = 0
        self.mass = defaultdict(lambda: [0.0, 0.0, 0.0])  # variant → [F, C, R]
        self.top_raise = 0.0   # max single raise prob (concentration probe)
        self.raise_total = 0.0

    def add(self, probs: list[float]):
        self.n += 1
        for name, kw in VARIANTS.items():
            p = sharpen(probs, **kw)
            self.mass[name][0] += p[0]
            self.mass[name][1] += p[1]
            self.mass[name][2] += sum(p[2:])
        self.top_raise += max(probs[2:]) if len(probs) > 2 else 0.0
        self.raise_total += sum(probs[2:])

    def row(self, variant: str) -> tuple[float, float, float]:
        if self.n == 0:
            return (0.0, 0.0, 0.0)
        f, c, r = self.mass[variant]
        return (f / self.n, c / self.n, r / self.n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="cmake-build-release/poker_ppo")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--out", default=None, help="JSONL decision log")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    eng = Engine(args.bin, args.model)
    out = open(args.out, "w") if args.out else None

    # (round_kind, facing, bucket) → Agg.  round_kind: preflop|postflop.
    aggs: dict[tuple, Agg] = defaultdict(Agg)
    decisions = 0

    try:
        for h in range(args.hands):
            st = eng.reset()
            while not st.done:
                m = eng.model()
                probs = m["probs"]
                hole = st.hole_p0 if st.cur_player == 0 else st.hole_p1
                if st.round == 0:
                    bucket = preflop_bucket(hole[0], hole[1])
                    facing = "vs-raise" if st.raises > 0 else "unopened"
                    kind = "preflop"
                else:
                    bucket = postflop_bucket(st.obs, st.round)
                    facing = "vs-bet" if st.cb > 0 else "unbet"
                    kind = "postflop"

                aggs[(kind, facing, bucket)].add(probs)
                decisions += 1
                if out:
                    out.write(json.dumps({
                        "hand": h, "round": st.round, "facing": facing,
                        "bucket": bucket, "probs": probs, "mask": st.mask,
                        "pot": st.pot, "cb": st.cb,
                    }) + "\n")

                st = eng.step(m["sampled"])
            if (h + 1) % 250 == 0:
                print(f"  ... {h + 1} hands, {decisions} decisions",
                      file=sys.stderr)
    finally:
        if out:
            out.close()
        eng.close()

    # ── Report ───────────────────────────────────────────────────────────
    def table(kind: str, facing: str, buckets: list[str]):
        print(f"\n═══ {kind} · {facing} ═══")
        hdr = f"  {'bucket':<10} {'n':>5} │"
        for v in VARIANTS:
            hdr += f"  {v:>9}: F /  C /  R │"
        print(hdr)
        for b in buckets:
            a = aggs.get((kind, facing, b))
            if not a or a.n < 20:
                continue
            line = f"  {b:<10} {a.n:>5} │"
            for v in VARIANTS:
                f, c, r = a.row(v)
                line += f"  {f*100:5.1f}/{c*100:5.1f}/{r*100:5.1f} │"
            conc = (a.top_raise / a.raise_total) if a.raise_total > 0 else 0.0
            line += f"  top-raise share {conc*100:4.1f}%"
            print(line)

    print(f"\npolicy audit: {args.hands} hands, {decisions} decisions")
    print("F/C/R = mean policy mass (%) on fold / call / all raises")
    print("top-raise share = largest single raise's share of total raise mass")
    print("  (high ⇒ concentrated belief; low ⇒ mass thinly spread = QRE noise)")
    for facing in ("unopened", "vs-raise"):
        table("preflop", facing, PRE_BUCKETS)
    for facing in ("unbet", "vs-bet"):
        table("postflop", facing, POST_BUCKETS)


if __name__ == "__main__":
    sys.exit(main() or 0)
