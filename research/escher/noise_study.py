"""
Noise study — WHICH regret-accumulation rule survives value-function error?

Motivation (HUNL findings): both ESCHER regret modes floor at LBR ~2.5-5
bb/hand while PPO reaches 1.85. The shared component is the privileged MC
value net: the sampled regret r = Q - V-bar inherits its error. This study
reproduces the HUNL trainer's exact loop semantics on Leduc with an EXACT
value oracle plus INJECTED error, isolating the sigma-dynamics question from
net-fitting confounds:

  - outcome-sampled visitation (traverser ~ uniform, opponent ~ sigma)
  - per-iteration MEAN-of-visits regret fit (what a net fit of the iter
    buffer computes), applied at iteration end
  - linear (weight t) sampled average strategy, judged by exact BR

Error model, mimicking a fitted critic:
  q_seen(s,a) = q_true(h,a) + persistent(s,a) + iid * N(0,1)
  - persistent: N(0, sp^2) per (infoset, action), redrawn every 10 iters
    (critic refit drift). This is the component that CANNOT average out.
  - iid: fresh per query (MC/minibatch noise). Averages out over visits.

Rules (all exist or are candidates for the HUNL trainer):
  rm+        cum <- max(0, cum + r)                 [neural-cum, ESCHER_GAMMA=1]
  dcfr.99    cum <- max(0, 0.99 cum + r)            [ESCHER_GAMMA=0.99]
  ema.9      play/avg sigma from EMA of cum         [ESCHER_REGRET_EMA=0.9]
  pred1      play RM+(cum + 1*(q - vbar_base))      [ESCHER_PREDICTIVE=1]
  paper      signed linear buffer: num += t*sum(r)  [ESCHER_REGRET_BUFFER=1]
  clip2      rm+ with r clipped to +-2*RMS(r)       [candidate, not in trainer]

Run: python3 noise_study.py [round1|round2] [iters]   (parallel, ~minutes)

RESULTS (1000 iters, Leduc, 2026-07-02) — final exploitability:

  round 1 (bias redrawn every 10 iters ~ critic refit drift):
    rule      clean   iid1.5  persist1.5  mixed.75
    rm+       0.149   0.205   0.853       0.401
    dcfr.99   0.347   0.380   1.191       0.753
    ema.9     0.213   0.238   0.537       0.422
    pred1     2.809   2.283   2.871       2.569
    paper     0.315   0.375   1.507(best 0.833 then DEGRADES)  0.915
    clip2     0.175   0.238   0.811       0.501

  round 2: permanent bias (never redrawn) = ~3.0-3.4 for EVERY rule —
  catastrophic and untreatable by sigma-dynamics. Critic-Polyak (qema=3)
  is a wash. sigma-EMA beta=.9 beats .95/.98 (lag dominates).

  round 3 (3000 iters, ema.9 base, gentle discounts, 2026-07-04): gamma<1
  hurts at EVERY strength (noise final: gamma1 0.17, .9995 0.21, .999 0.27)
  and NO late creep appears in this chassis even at 3000 iters — the HUNL
  cur-LBR creep is therefore a NET-fitting phenomenon (warm regret-net drift
  / sigma-sharpening coverage feedback), not an accumulation-rule property.
  Harvest best-iterates (cur_best.pt); don't discount.

CONCLUSIONS for the HUNL trainer:
  1. iid value noise is benign; PERSISTENT critic bias sets the floor.
     Fix the critic, not the sampler: ESCHER_VALUE_EPS>0 (converts the
     catastrophic never-supervised-action bias into small drifting bias),
     ESCHER_LAMBDA~0.5, ESCHER_VALUE_BUF_CAP>0, more value capacity/steps.
  2. ESCHER_REGRET_EMA=0.9 is the only sigma-dynamics lever that helps
     (0.85 -> 0.54 under drifting bias) — keep it on.
  3. ESCHER_GAMMA<1 HURTS under bias (shorter memory = more dependence on
     the currently-biased regrets). Keep 1.0.
  4. ESCHER_PREDICTIVE is harmful even with EXACT values in the sampled
     chassis (2.8 vs 0.15 clean) — do not use on HUNL as implemented.
  5. paper buffer mode reproduces the HUNL stall-then-degrade signature
     under drifting bias; neural-cum RM+ dominates it everywhere.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from games import Leduc
from solvers import exploitability


# ─── rules ───────────────────────────────────────────────────────────────────
class Rule:
    def __init__(self, name, gamma=1.0, ema=0.0, pred=0.0, paper=False, clip=0.0):
        self.name, self.gamma, self.ema_b, self.pred = name, gamma, ema, pred
        self.paper, self.clip_k = paper, clip
        self.cum = defaultdict(lambda: defaultdict(float))   # true cumulative
        self.ema = defaultdict(lambda: defaultdict(float))   # EMA shadow
        self.rms = 1.0                                        # running RMS of |r|

    def _rmplus(self, table, I, legal):
        pos = {a: max(table[I][a], 0.0) for a in legal}
        s = sum(pos.values())
        if s > 1e-12:
            return {a: pos[a] / s for a in legal}
        return {a: 1.0 / len(legal) for a in legal}

    # played sigma at a node. q_seen = the (noisy) oracle row, needed by pred.
    def play(self, I, legal, q_seen):
        src = self.ema if self.ema_b > 0 else self.cum
        if self.pred <= 0:
            return self._rmplus(src, I, legal)
        base = self._rmplus(src, I, legal)
        vbar = sum(base[a] * q_seen[a] for a in legal)
        pos = {a: max(src[I][a] + self.pred * (q_seen[a] - vbar), 0.0)
               for a in legal}
        s = sum(pos.values())
        if s > 1e-12:
            return {a: pos[a] / s for a in legal}
        return {a: 1.0 / len(legal) for a in legal}

    # end-of-iteration fit from {I: [rvec per visit]} (mean = the net fit)
    def update(self, iter_acc, t):
        sq, n = 0.0, 0
        for I, rvecs in iter_acc.items():
            legal = list(rvecs[0].keys())
            mean_r = {a: sum(rv[a] for rv in rvecs) / len(rvecs) for a in legal}
            for a in legal:
                sq += mean_r[a] ** 2
                n += 1
            if self.paper:  # idealized reservoir fit: iteration-weighted mean
                for a in legal:
                    self.cum[I][a] += t * sum(rv[a] for rv in rvecs)
                continue
            c = self.clip_k * self.rms if self.clip_k > 0 else 0.0
            for a in legal:
                r = mean_r[a]
                if c > 0.0:
                    r = max(-c, min(c, r))
                self.cum[I][a] = max(0.0, self.gamma * self.cum[I][a] + r)
        if n:
            self.rms = 0.9 * self.rms + 0.1 * math.sqrt(sq / n)
        if self.ema_b > 0:
            b = self.ema_b
            keys = set(self.cum) | set(self.ema)
            for I in keys:
                for a in set(self.cum[I]) | set(self.ema[I]):
                    self.ema[I][a] = b * self.ema[I][a] + (1 - b) * self.cum[I][a]


# ─── one study cell ──────────────────────────────────────────────────────────
def run_cell(args):
    rule_spec, sp, si, iters, seed, redraw, qema = args
    g = Leduc
    rng = random.Random(seed)
    rule = Rule(**rule_spec)
    deals = g.deals()
    strat_sum = defaultdict(lambda: defaultdict(float))
    n_traj = 128
    # persist[k] = list of per-epoch draws (newest last); the bias the agent
    # SEES is the mean of the last `qema` epochs — a Polyak/EMA-averaged
    # critic (qema=1 → live critic, the current trainer behaviour).
    persist = {}
    persist_epoch = -1

    def pers(I, a):
        if sp == 0.0:
            return 0.0
        k = (I, a)
        hist = persist.get(k)
        if hist is None or len(hist) <= persist_epoch:
            hist = persist.setdefault(k, [])
            while len(hist) <= persist_epoch:
                hist.append(rng.gauss(0.0, sp))
        recent = hist[-qema:] if len(hist) >= qema else hist
        return sum(recent) / len(recent)

    def sample_deal():
        r, acc = rng.random(), 0.0
        for cards, p in deals:
            acc += p
            if r <= acc:
                return cards
        return deals[-1][0]

    def choose(legal, sig):
        r, acc = rng.random(), 0.0
        for a in legal:
            acc += sig[a]
            if r <= acc:
                return a
        return legal[-1]

    # exact q under the PLAYED policy: memo[(cards,h)] = {a: q_true_p0}.
    # sigma inside uses the persistent-noise view (the policy the agent
    # actually implements); returned q values are exact.
    def build_memo():
        memo = {}

        def val(h, cards):
            if g.is_terminal(h):
                return g.terminal_util_p0(h, cards)
            key = (cards, h)
            if key in memo:
                q = memo[key]
            else:
                legal = g.legal_actions(h)
                q = {a: val(g.step_history(h, a), cards) for a in legal}
                memo[key] = q
            p = g.current_player(h)
            I = g.infoset_key(h, cards)
            legal = list(q.keys())
            sign = 1.0 if p == 0 else -1.0
            q_seen = {a: sign * q[a] + pers(I, a) for a in legal}
            sig = rule.play(I, legal, q_seen)
            return sum(sig[a] * q[a] for a in legal)

        seen = set()
        for cards, _ in deals:
            if cards in seen:
                continue
            seen.add(cards)
            val("", cards)
        return memo

    evals = []
    for t in range(1, iters + 1):
        persist_epoch = (t - 1) // redraw
        memo = build_memo()

        def q_row(h, cards, I, legal, p):  # acting-player frame, noisy
            sign = 1.0 if p == 0 else -1.0
            q = memo[(cards, h)]
            return {a: sign * q[a] + pers(I, a) +
                       (si * rng.gauss(0.0, 1.0) if si > 0.0 else 0.0)
                    for a in legal}

        # regret phase: fixed sampler, per-iteration accumulator
        iter_acc = defaultdict(list)
        for traverser in (0, 1):
            for _ in range(n_traj):
                cards = sample_deal()
                h = ""
                while not g.is_terminal(h):
                    p = g.current_player(h)
                    legal = g.legal_actions(h)
                    I = g.infoset_key(h, cards)
                    qs = q_row(h, cards, I, legal, p)
                    sig = rule.play(I, legal, qs)
                    if p == traverser:
                        vbar = sum(sig[a] * qs[a] for a in legal)
                        iter_acc[I].append({a: qs[a] - vbar for a in legal})
                        a = legal[rng.randrange(len(legal))]
                    else:
                        a = choose(legal, sig)
                    h = g.step_history(h, a)
        rule.update(iter_acc, t)

        # average phase under the played sigma, weight t
        for _ in range(n_traj):
            cards = sample_deal()
            h = ""
            while not g.is_terminal(h):
                p = g.current_player(h)
                legal = g.legal_actions(h)
                I = g.infoset_key(h, cards)
                qs = (q_row(h, cards, I, legal, p) if rule.pred > 0
                      else {a: 0.0 for a in legal})
                sig = rule.play(I, legal, qs)
                a = choose(legal, sig)
                strat_sum[I][a] += t
                h = g.step_history(h, a)

        if t % 50 == 0 or t == iters:
            evals.append((t, expl_of(g, deals, strat_sum)))

    return rule_spec["name"], sp, si, redraw, qema, evals


def expl_of(g, deals, strat_sum):
    # full-coverage average strategy (uniform where never visited)
    out, seen = {}, set()

    def rec(h, cards):
        if g.is_terminal(h):
            return
        I = g.infoset_key(h, cards)
        legal = g.legal_actions(h)
        if I not in out:
            ss = strat_sum.get(I)
            tot = sum(ss[a] for a in legal) if ss else 0.0
            out[I] = ({a: ss[a] / tot for a in legal} if tot > 0
                      else {a: 1.0 / len(legal) for a in legal})
        for a in legal:
            rec(g.step_history(h, a), cards)

    for cards, _ in deals:
        if cards in seen:
            continue
        seen.add(cards)
        rec("", cards)
    return exploitability(g, out)


RULES = [
    dict(name="rm+"),
    dict(name="dcfr.99", gamma=0.99),
    dict(name="ema.9", ema=0.9),
    dict(name="pred1", pred=1.0),
    dict(name="paper", paper=True),
    dict(name="clip2", clip=2.0),
]
# (persistent sd, iid sd) in chips; Leduc value-net V-rmse was measured ~3
NOISE = [(0.0, 0.0), (0.0, 1.5), (1.5, 0.0), (0.75, 0.75)]
FIXED = 10 ** 9   # redraw interval = never (permanent bias)


def round1_cells(iters):
    # (rule, sp, si, iters, seed, redraw, qema)
    return [(r, sp, si, iters, 7, 10, 1) for r in RULES for (sp, si) in NOISE]


def round2_cells(iters):
    # σ-EMA depth × critic-EMA (qema) under drifting AND permanent bias.
    rules = [dict(name="rm+"), dict(name="ema.9", ema=0.9),
             dict(name="ema.95", ema=0.95), dict(name="ema.98", ema=0.98)]
    cells = []
    for r in rules:
        for qema in (1, 3):
            cells.append((r, 1.5, 0.0, iters, 7, 10, qema))      # drifting
            cells.append((r, 1.5, 0.0, iters, 7, FIXED, qema))   # permanent
            cells.append((r, 0.75, 0.75, iters, 7, 10, qema))    # mixed
    return cells


def round3_cells(iters):
    # Late-creep control: with sigma near Nash, true regrets shrink and the
    # gamma=1 cumulative integrates rectified noise (observed on HUNL: cur-LBR
    # creeps up past ~iter 2300, rmag grows monotonically). Does a GENTLE
    # discount (0.999/0.9995 — round 1's 0.99 was too aggressive) bound the
    # integral without hurting the floor? All rules on top of ema.9 (prod).
    rules = [dict(name="ema.9", ema=0.9),
             dict(name="g.9995", ema=0.9, gamma=0.9995),
             dict(name="g.999", ema=0.9, gamma=0.999)]
    cells = []
    for r in rules:
        cells.append((r, 0.5, 0.5, iters, 7, 10, 1))   # mild drifting bias
        cells.append((r, 0.0, 0.0, iters, 7, 10, 1))   # clean control
    return cells


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "round1"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cells = (round1_cells(iters) if which == "round1"
             else round2_cells(iters) if which == "round2"
             else round3_cells(iters))
    with ProcessPoolExecutor(max_workers=7) as ex:
        for name, sp, si, redraw, qema, evals in ex.map(run_cell, cells):
            final = evals[-1][1]
            best = min(e for _, e in evals)
            tag = "fixed" if redraw == FIXED else f"rd{redraw}"
            print(f"  {name:8s} sp={sp:4.2f} si={si:4.2f} {tag:6s} qema={qema}"
                  f"  final={final:.4f}  best={best:.4f}", flush=True)
            if which == "round3":   # creep needs the trajectory, not extrema
                traj = "   " + " ".join(f"{t}:{e:.2f}" for t, e in evals)
                print(traj, flush=True)


if __name__ == "__main__":
    main()
