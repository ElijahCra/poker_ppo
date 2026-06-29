"""
Stage 2: NEURAL ESCHER (Deep-CFR-style) on Kuhn-N / Leduc.

Validates that ESCHER converges with FUNCTION APPROXIMATION + reservoir
buffers (the real risk before the C++/HUNL port), judged by the exact
recursive BR (solvers.exact_br_value).

Two modes (the staging):
  values='exact'  — action values from exact recursion (Stage 2a): isolates
                    the NN regret/strategy/buffer pipeline.
  values='net'    — action values from a LEARNED history-value net trained
                    IS-free on sampled trajectories (Stage 2b): the full
                    ESCHER, the part that ports to HUNL.

Networks (all tiny MLPs):
  regret_net(infoset_feat) -> per-action cumulative-regret estimate; σ = RM⁺.
  avg_net(infoset_feat)    -> per-action logits; the DEPLOYED average policy.
  value_net(history_feat)  -> scalar P0-value under σ (mode 'net' only).

Run in WSL: python3 escher_nn.py [kuhn6|kuhn|leduc] [exact|net]
"""

from __future__ import annotations

import copy
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from games import Kuhn, Leduc, make_kuhn_n
from solvers import exploitability, expected_value

DEV = "cpu"  # nets are tiny; CPU avoids launch overhead on small batches


# ─── feature encoding ───────────────────────────────────────────────────────
class Featurizer:
    """Infoset/history -> fixed feature vector. Card as a normalised scalar
    AND a one-hot (forces some generalisation over rank while staying
    separable); public one-hot (Leduc); betting history one-hot."""

    def __init__(self, game, n_ranks: int):
        self.g = game
        self.n_ranks = n_ranks
        self.has_public = any("/" in self._hist_of(k)
                              for k in self._all_infoset_keys())
        hists = sorted({self._hist_of(k) for k in self._all_infoset_keys()})
        self.hidx = {h: i for i, h in enumerate(hists)}
        self.n_hist = len(hists)
        # infoset feat: card_scalar(1) + card_onehot(R) + public_onehot(R+1
        #  if public else 0) + hist_onehot(H) + player(1)
        self.pub_dim = (n_ranks + 1) if self.has_public else 0
        self.inf_dim = 1 + n_ranks + self.pub_dim + self.n_hist + 1
        # full history feat (value net): both cards + public + hist + showdown
        # interaction features (pair flags + win/tie/lose) — additive one-hots
        # express the card0×card1×public showdown conjunction poorly, and under
        # a sharp σ the value IS ~ the showdown outcome. Mirrors the
        # hand-strength features a real (HUNL) value net is given.
        self.extra_dim = 3 + (2 if self.has_public else 0)
        self.full_dim = 2 * n_ranks + self.pub_dim + self.n_hist + self.extra_dim

    def _all_infoset_keys(self):
        keys = set()
        seen = set()

        def rec(h, cards):
            if self.g.is_terminal(h):
                return
            keys.add(self.g.infoset_key(h, cards))
            for a in self.g.legal_actions(h):
                rec(self.g.step_history(h, a), cards)

        for cards, _ in self.g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        return keys

    @staticmethod
    def _hist_of(key: str) -> str:
        return key.split("|")[-1]

    def _public_idx(self, history, cards):
        # 0 = not revealed; else 1+rank
        if not self.has_public:
            return 0
        return (1 + cards[2]) if "/" in history else 0

    def infoset(self, history, cards, player):
        v = torch.zeros(self.inf_dim)
        i = 0
        card = cards[player]
        v[i] = card / max(1, self.n_ranks - 1); i += 1
        v[i + card] = 1.0; i += self.n_ranks
        if self.has_public:
            v[i + self._public_idx(history, cards)] = 1.0; i += self.pub_dim
        v[i + self.hidx[history]] = 1.0; i += self.n_hist
        v[i] = float(player)
        return v

    def full(self, history, cards):
        # PRIVILEGED full-state value features: the value net conditions on the
        # whole deal, so it ALWAYS sees the public card (even in round 1, where
        # players don't). Hiding it aliases round-1 states with different public
        # cards — fine under near-uniform σ, but ruinous under a sharp σ where
        # the round-1 value depends critically on the showdown-determining card.
        v = torch.zeros(self.full_dim)
        i = 0
        v[i + cards[0]] = 1.0; i += self.n_ranks
        v[i + cards[1]] = 1.0; i += self.n_ranks
        if self.has_public:
            v[i + 1 + cards[2]] = 1.0; i += self.pub_dim  # always revealed
        v[i + self.hidx[history]] = 1.0; i += self.n_hist
        # showdown interaction features
        if self.has_public:
            pub = cards[2]
            p0p, p1p = (cards[0] == pub), (cards[1] == pub)
            v[i] = float(p0p); v[i + 1] = float(p1p); i += 2
            r0 = (self.n_ranks + cards[0]) if p0p else cards[0]
            r1 = (self.n_ranks + cards[1]) if p1p else cards[1]
        else:
            r0, r1 = cards[0], cards[1]
        v[i + (0 if r0 > r1 else 1 if r0 == r1 else 2)] = 1.0  # win/tie/lose
        return v


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class Reservoir:
    def __init__(self, cap, seed):
        self.cap = cap
        self.data = []
        self.n = 0
        self.rng = random.Random(seed)

    def add(self, item):
        self.n += 1
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            j = self.rng.randint(0, self.n - 1)
            if j < self.cap:
                self.data[j] = item


# ─── Neural ESCHER ──────────────────────────────────────────────────────────
class NeuralESCHER:
    def __init__(self, game, n_ranks, values="exact",
                 hidden=64, reg_steps=400, buf_cap=200_000, seed=0,
                 val_hidden=256):
        self.g = game
        self.feat = Featurizer(game, n_ranks)
        self.values_mode = values
        self.A = game.n_actions
        self.hidden = hidden
        self.val_hidden = val_hidden
        self.reg_steps = reg_steps
        self.t = 0  # CFR iteration counter (survives checkpoint/resume)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.regret_net = MLP(self.feat.inf_dim, self.A, hidden).to(DEV)
        self.avg_net = MLP(self.feat.inf_dim, self.A, hidden).to(DEV)
        if values == "net":
            self.value_net = MLP(self.feat.full_dim, 1, val_hidden).to(DEV)
            self.value_opt = torch.optim.Adam(self.value_net.parameters(), 1e-3)
            self.val_buf = Reservoir(buf_cap, seed + 3)
            self.v_scale = self._compute_v_scale()  # net predicts value/v_scale
            self.v_expl = 0.6  # sampling = expl·uniform + (1-expl)·σ (coverage)
            self.val_sweeps = 4  # fitted-value sweeps per CFR iter (track σ)
        self.reg_buf = Reservoir(buf_cap, seed + 1)
        self.strat_buf = Reservoir(buf_cap, seed + 2)
        # exact tabular linear-average accumulator — DIAGNOSTIC only, to tell
        # whether the avg-NET readout or the core regret dynamics is the limit.
        self.tab_ss = defaultdict(lambda: [0.0] * self.A)
        self.tab_w = defaultdict(float)
        self.tab_legal = {}
        # exact-cumulative mode: maintain the EXACT RM⁺ cumulative regret per
        # infoset (full-tree gives it directly), regress the net to THAT instead
        # of a noisy sampled buffer — isolates FA representational adequacy
        # (σ still comes from the net, so FA stays in the CFR loop).
        self.exact_cum = False
        self.cum = defaultdict(lambda: [0.0] * self.A)
        self.cum_legal = {}
        self._cum_meta = {}

    # current strategy at an infoset via regret-matching⁺ on the regret net
    @torch.no_grad()
    def sigma(self, history, cards, player, legal):
        f = self.feat.infoset(history, cards, player).to(DEV)
        r = self.regret_net(f.unsqueeze(0)).squeeze(0)
        pos = torch.clamp(r, min=0.0)
        mask = torch.full((self.A,), 0.0)
        for a in legal:
            mask[a] = 1.0
        pos = pos * mask
        s = pos.sum().item()
        if s > 1e-12:
            p = (pos / s).tolist()
            return {a: p[a] for a in legal}
        return {a: 1.0 / len(legal) for a in legal}

    # ── exact σ-value of a history (recursion) — used by 'exact' mode and as
    #    the value-net bootstrap target in 'net' mode ──
    def exact_value(self, history, cards):
        g = self.g
        if g.is_terminal(history):
            return g.terminal_util_p0(history, cards)
        player = g.current_player(history)
        legal = g.legal_actions(history)
        sig = self.sigma(history, cards, player, legal)
        return sum(sig[a] * self.exact_value(g.step_history(history, a), cards)
                   for a in legal)

    def _compute_v_scale(self):
        g = self.g
        mx = [1.0]

        def rec(h, cards):
            if g.is_terminal(h):
                mx[0] = max(mx[0], abs(g.terminal_util_p0(h, cards)))
                return
            for a in g.legal_actions(h):
                rec(g.step_history(h, a), cards)

        seen = set()
        for cards, _ in g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        return mx[0]

    @torch.no_grad()
    def net_value(self, history, cards):
        if self.g.is_terminal(history):
            return self.g.terminal_util_p0(history, cards)
        f = self.feat.full(history, cards).to(DEV)
        return self.value_net(f.unsqueeze(0)).item() * self.v_scale

    def _choose_sigma(self, legal, sig):
        r = self.rng.random()
        acc = 0.0
        for a in legal:
            acc += sig[a]
            if r <= acc:
                return a
        return legal[-1]

    @torch.no_grad()
    def value_diag(self):
        """Reach-weighted RMSE (and max) of net_value vs the EXACT on-σ history
        value — the value net's actual job. One post-order pass; the direct
        signal for tuning the IS-free value learning without full convergence."""
        g = self.g
        acc = {"se": 0.0, "w": 0.0, "mx": 0.0}

        def rec(h, cards, reach):
            if g.is_terminal(h):
                return g.terminal_util_p0(h, cards)
            player = g.current_player(h)
            legal = g.legal_actions(h)
            sig = self.sigma(h, cards, player, legal)
            ev = 0.0
            for a in legal:
                ev += sig[a] * rec(g.step_history(h, a), cards, reach * sig[a])
            nv = self.net_value(h, cards)
            acc["se"] += reach * (nv - ev) ** 2
            acc["w"] += reach
            acc["mx"] = max(acc["mx"], abs(nv - ev))
            return ev

        for cards, p in g.deals():
            rec("", cards, p)
        return (acc["se"] / max(acc["w"], 1e-9)) ** 0.5, acc["mx"]

    def value_of(self, history, cards):
        return (self.net_value if self.values_mode == "net"
                else self.exact_value)(history, cards)

    # ── train the value net IS-free: bootstrap visited states toward the on-σ
    #    expected child value. A REPLAY BUFFER of (history, cards) states
    #    visited under the fixed (uniform) sampling policy accumulates coverage
    #    across iterations; each step samples a minibatch and recomputes the
    #    bootstrap target Σσ(a)V(child) with the CURRENT net+σ (off-policy
    #    samples, on-σ targets, NO importance weight — the ESCHER trick). Far
    #    more stable than re-fitting fresh trajectories each iteration. ──
    def train_value(self, n_traj=64, steps=120, n_states=2048):
        g = self.g
        # grow coverage with an expl-mixed (uniform/σ) sampling policy — pure
        # uniform under-visits the on-path states the regret actually needs.
        for _ in range(n_traj):
            cards = self._sample_deal()
            h = ""
            while not g.is_terminal(h):
                self.val_buf.add((h, cards))
                legal = g.legal_actions(h)
                if self.rng.random() < self.v_expl:
                    a = self.rng.choice(legal)
                else:
                    p = g.current_player(h)
                    a = self._choose_sigma(legal, self.sigma(h, cards, p, legal))
                h = g.step_history(h, a)
        data = self.val_buf.data
        if not data:
            return
        # frozen-target fitted value iteration: σ and the bootstrap targets are
        # FIXED for this iteration (snapshot net), so build (feat, target) ONCE
        # then take many cheap full-batch steps — stable AND fast.
        target_net = copy.deepcopy(self.value_net)

        @torch.no_grad()
        def tval(h, c):
            if g.is_terminal(h):
                return g.terminal_util_p0(h, c)
            f = self.feat.full(h, c).to(DEV)
            return target_net(f.unsqueeze(0)).item() * self.v_scale

        idx = [self.rng.randrange(len(data))
               for _ in range(min(n_states, len(data)))]
        feats, targets = [], []
        for i in idx:
            hist, c = data[i]
            legal = g.legal_actions(hist)
            p = g.current_player(hist)
            sig = self.sigma(hist, c, p, legal)
            tgt = sum(sig[a] * tval(g.step_history(hist, a), c) for a in legal)
            feats.append(self.feat.full(hist, c))
            targets.append(tgt / self.v_scale)  # net predicts normalised value
        X = torch.stack(feats).to(DEV)
        y = torch.tensor(targets, dtype=torch.float32).to(DEV).unsqueeze(1)
        for _ in range(steps):
            self.value_opt.zero_grad()
            loss = F.mse_loss(self.value_net(X), y)
            loss.backward()
            self.value_opt.step()

    # ── full-tree regret pass: instantaneous counterfactual regret per
    #    infoset (summed over deals), action values from value_of(). ──
    def collect_regret(self, t):
        g = self.g
        inst = defaultdict(lambda: [0.0] * self.A)
        strat = {}
        key_meta = {}

        def rec(history, cards, p0, p1, q):
            if g.is_terminal(history):
                return g.terminal_util_p0(history, cards)
            player = g.current_player(history)
            legal = g.legal_actions(history)
            I = g.infoset_key(history, cards)
            sig = self.sigma(history, cards, player, legal)
            util_a = {}
            node = 0.0
            for a in legal:
                util_a[a] = rec(g.step_history(history, a), cards,
                                p0 * (sig[a] if player == 0 else 1.0),
                                p1 * (sig[a] if player == 1 else 1.0), q)
                node += sig[a] * util_a[a]
            # q = chance/deal reach: π_{-i} and the average weight both include
            # chance, so scale by the deal probability (silent on uniform-deal
            # Kuhn; essential on Leduc's non-uniform deck).
            cf = q * (p1 if player == 0 else p0)
            own = q * (p0 if player == 0 else p1)
            sign = 1.0 if player == 0 else -1.0
            # value-function action values for the regret (ESCHER): replace
            # the recursion utils with value_of(child) so 'net' mode is IS-free.
            for a in legal:
                if self.values_mode == "net":
                    va = self.value_of(g.step_history(history, a), cards)
                    vn = self.value_of(history, cards)
                else:
                    va, vn = util_a[a], node
                inst[I][a] += cf * sign * (va - vn)
            strat.setdefault(I, [0.0] * self.A)
            for a in legal:
                strat[I][a] += own * sig[a]
            key_meta[I] = (history, cards, player, legal)
            return node

        for cards, p in g.deals():
            rec("", cards, 1.0, 1.0, p)

        self._inst = inst          # for the neural-cumulative fitter
        self._key_meta = key_meta
        for I, rvec in inst.items():
            history, cards, player, legal = key_meta[I]
            f = self.feat.infoset(history, cards, player)
            self.reg_buf.add((f, torch.tensor(rvec), float(t)))
            if self.exact_cum:  # RM⁺: floor the cumulative regret at 0
                for a in legal:
                    self.cum[I][a] = max(0.0, self.cum[I][a] + rvec[a])
                self.cum_legal[I] = legal
                self._cum_meta[I] = (history, cards, player)
        for I, svec in strat.items():
            history, cards, player, legal = key_meta[I]
            tot = sum(svec)
            if tot > 0:
                probs = [x / tot for x in svec]
                f = self.feat.infoset(history, cards, player)
                self.strat_buf.add((f, torch.tensor(probs), float(t)))
                # mirror the buffer target into the exact tabular average
                for a in range(self.A):
                    self.tab_ss[I][a] += t * probs[a]
                self.tab_w[I] += t
                self.tab_legal[I] = legal

    def tabular_average(self):
        """The exact linear-weighted average of the same σ targets the avg-net
        is trained on — the ceiling the net could reach with a perfect fit.
        Enumerates the whole tree (BR needs every infoset); unvisited infosets
        (zeroed-out by RM⁺ reach) fall back to uniform."""
        out = {}
        seen = set()
        g = self.g

        def rec(h, cards):
            if g.is_terminal(h):
                return
            I = g.infoset_key(h, cards)
            legal = g.legal_actions(h)
            if I not in out:
                if self.tab_w.get(I, 0.0) > 0:
                    w = self.tab_w[I]
                    out[I] = {a: self.tab_ss[I][a] / w for a in legal}
                else:
                    out[I] = {a: 1.0 / len(legal) for a in legal}
            for a in legal:
                rec(g.step_history(h, a), cards)

        for cards, _ in g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        return out

    def _train_net(self, net, buf, steps, lr, mb=512, normalize=False):
        opt = torch.optim.Adam(net.parameters(), lr)
        data = buf.data
        if not data:
            return
        # global target scale: RM is scale-invariant per infoset, so dividing
        # ALL regret targets by one positive constant leaves σ unchanged but
        # turns the ±13-magnitude Leduc regrets into O(1) targets the MSE fits
        # far more accurately — directly lowers the FA noise floor.
        scale = 1.0
        if normalize:
            ss = torch.stack([data[i][1] for i in
                              self.rng.sample(range(len(data)),
                                              min(4096, len(data)))])
            scale = max(1e-6, ss.pow(2).mean().sqrt().item())
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(mb, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            Y = torch.stack([data[i][1] for i in idx]).to(DEV) / scale
            W = torch.tensor([data[i][2] for i in idx]).to(DEV).unsqueeze(1)
            opt.zero_grad()
            pred = net(X)
            loss = (W * (pred - Y) ** 2).mean()
            loss.backward()
            opt.step()

    def fit_regret(self, lr=3e-3):
        # neural-cumulative (SCALABLE): the regret NET itself stores the running
        # RM⁺ cumulative regret — no per-infoset table, smooth σ. Must NOT
        # fresh-init (the net IS the accumulator).
        if getattr(self, "neural_cum", False):
            self._fit_regret_neural_cum(lr)
            return
        # Deep CFR fresh-inits each iter (unbiased), but that injects random
        # reinit jitter into σ_t every iteration → the average can't tighten.
        # warm-start keeps the net and nudges it toward the updated buffer →
        # SMOOTH σ trajectory → lower averaging floor (empirically better on
        # small setups, the trade-off being a little stale-target bias).
        if not getattr(self, "warm", False):
            self.regret_net = MLP(self.feat.inf_dim, self.A, self.hidden).to(DEV)
        if self.exact_cum:
            self._fit_regret_cum(lr)
        else:
            self._train_net(self.regret_net, self.reg_buf, self.reg_steps, lr,
                            normalize=True)

    def _fit_regret_neural_cum(self, lr):
        """SCALABLE analog of exact-cum: store the RM⁺ cumulative regret IN THE
        NET (warm-trained toward max(0, γ·prev_net(I) + inst(I))) instead of a
        per-infoset table. γ<1 discounts (bounds magnitude, recency-weights —
        DCFR-style). One stable target per infoset → smooth σ → value tracks."""
        items = list(self._key_meta.items())
        if not items:
            return
        feats = [self.feat.infoset(h, c, p) for _, (h, c, p, lg) in items]
        X = torch.stack(feats).to(DEV)
        prev = copy.deepcopy(self.regret_net)
        with torch.no_grad():
            prevR = prev(X)
        Y = torch.zeros_like(prevR)
        g = getattr(self, "ncum_gamma", 0.99)
        for idx, (I, (h, c, p, legal)) in enumerate(items):
            inst = self._inst[I]
            for a in legal:
                Y[idx, a] = max(0.0, g * prevR[idx, a].item() + inst[a])
        opt = torch.optim.Adam(self.regret_net.parameters(), lr)
        for _ in range(self.reg_steps):
            opt.zero_grad()
            loss = F.mse_loss(self.regret_net(X), Y)
            loss.backward()
            opt.step()

    def _fit_regret_cum(self, lr):
        """Regress the regret net to the EXACT RM⁺ cumulative regret table
        (one stable target per infoset) — the FA-adequacy probe."""
        feats, tgts = [], []
        for I, legal in self.cum_legal.items():
            history, cards, player = self._cum_meta[I]
            feats.append(self.feat.infoset(history, cards, player))
            tgts.append(torch.tensor(self.cum[I]))
        X = torch.stack(feats).to(DEV)
        Y = torch.stack(tgts).to(DEV)
        # PER-INFOSET normalisation: RM(I) is invariant to positive scaling of
        # R(I), so rescale each infoset's regret vector to unit norm. This
        # weights every infoset equally in the MSE (a global scale lets large-
        # regret infosets dominate, under-fitting the close-regret / finely-
        # mixed ones where RM is most error-sensitive) while preserving the
        # within-infoset ratios RM actually consumes.
        row = Y.norm(dim=1, keepdim=True).clamp_min(1e-6)
        Y = Y / row
        opt = torch.optim.Adam(self.regret_net.parameters(), lr)
        for _ in range(self.reg_steps):
            opt.zero_grad()
            loss = F.mse_loss(self.regret_net(X), Y)
            loss.backward()
            opt.step()

    def fit_avg(self, steps=600, lr=2e-3):
        self.avg_net = MLP(self.feat.inf_dim, self.A, self.hidden).to(DEV)
        # avg net regresses to the iteration-weighted average σ (target probs);
        # softmax at read time. Train logits via MSE on probs is crude but ok
        # for small games; use KL via log-softmax instead.
        opt = torch.optim.Adam(self.avg_net.parameters(), lr)
        data = self.strat_buf.data
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(1024, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            Y = torch.stack([data[i][1] for i in idx]).to(DEV)
            W = torch.tensor([data[i][2] for i in idx]).to(DEV)
            opt.zero_grad()
            logp = F.log_softmax(self.avg_net(X), dim=1)
            loss = (W * (-(Y * logp).sum(1))).mean()  # weighted cross-entropy
            loss.backward()
            opt.step()

    @torch.no_grad()
    def avg_strategy(self):
        """Extract the avg-net policy for every infoset as a tabular dict for
        exact exploitability."""
        out = {}
        seen = set()
        g = self.g

        def rec(h, cards):
            if g.is_terminal(h):
                return
            player = g.current_player(h)
            legal = g.legal_actions(h)
            I = g.infoset_key(h, cards)
            if I not in out:
                f = self.feat.infoset(h, cards, player).to(DEV)
                logits = self.avg_net(f.unsqueeze(0)).squeeze(0)
                m = torch.full((self.A,), float("-inf"))
                for a in legal:
                    m[a] = logits[a]
                p = F.softmax(m, dim=0).tolist()
                out[I] = {a: p[a] for a in legal}
            for a in legal:
                rec(g.step_history(h, a), cards)

        for cards, _ in g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        return out

    def _sample_deal(self):
        deals = self.g.deals()
        r = self.rng.random()
        acc = 0.0
        for cards, p in deals:
            acc += p
            if r <= acc:
                return cards
        return deals[-1][0]

    def run(self, iters, eval_every=10, ckpt=None):
        # tabular-avg is the HEADLINE: it's the exact linear-CFR average of the
        # σ sequence (≡ a perfectly-fit avg net, per the diagnostic), so it
        # measures how well the DYNAMICS converge — the floor being attacked.
        target = self.t + iters
        while self.t < target:
            self.t += 1
            eval_now = (self.t % eval_every == 0 or self.t == target)
            extra = ""
            if self.values_mode == "net":
                # several fitted-value sweeps so the value net CATCHES UP to the
                # current σ before the regret pass reads it (1 backup/iter lags a
                # moving σ; with fixed σ value learning hits ~0.04).
                for _ in range(self.val_sweeps):
                    self.train_value(n_traj=32, steps=80)
                if eval_now:  # measure for the σ collect_regret will USE
                    rmse, mx = self.value_diag()
                    extra = f"   V-rmse={rmse:.4f} (max {mx:.3f})"
            self.collect_regret(self.t)
            self.fit_regret()
            if eval_now:
                tab = exploitability(self.g, self.tabular_average())
                print(f"  iter {self.t:4d}   expl(tabular-avg)={tab:.5f}{extra}",
                      flush=True)
                if ckpt:
                    self.save(ckpt)
        return tab

    # ── checkpoint/resume: accumulate iterations across the 600s-per-call cap ──
    def save(self, path):
        torch.save({
            "t": self.t,
            "regret_net": self.regret_net.state_dict(),
            "reg_buf": self.reg_buf.data, "reg_n": self.reg_buf.n,
            "strat_buf": self.strat_buf.data, "strat_n": self.strat_buf.n,
            "tab_ss": dict(self.tab_ss), "tab_w": dict(self.tab_w),
            "tab_legal": self.tab_legal,
            "value_net": (self.value_net.state_dict()
                          if self.values_mode == "net" else None),
            "val_buf": (self.val_buf.data if self.values_mode == "net" else None),
            "val_n": (self.val_buf.n if self.values_mode == "net" else 0),
            "cum": dict(self.cum), "cum_legal": self.cum_legal,
            "cum_meta": self._cum_meta,
        }, path)

    def load(self, path):
        d = torch.load(path, weights_only=False)
        self.t = d["t"]
        self.regret_net.load_state_dict(d["regret_net"])
        self.reg_buf.data = d["reg_buf"]; self.reg_buf.n = d["reg_n"]
        self.strat_buf.data = d["strat_buf"]; self.strat_buf.n = d["strat_n"]
        self.tab_ss = defaultdict(lambda: [0.0] * self.A, d["tab_ss"])
        self.tab_w = defaultdict(float, d["tab_w"])
        self.tab_legal = d["tab_legal"]
        if self.values_mode == "net" and d["value_net"]:
            self.value_net.load_state_dict(d["value_net"])
            self.val_buf.data = d["val_buf"]; self.val_buf.n = d["val_n"]
        if d.get("cum"):
            self.cum = defaultdict(lambda: [0.0] * self.A, d["cum"])
            self.cum_legal = d["cum_legal"]
            self._cum_meta = d["cum_meta"]


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="kuhn6")
    ap.add_argument("mode", nargs="?", default="exact")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--reg-steps", type=int, default=400)
    ap.add_argument("--val-hidden", type=int, default=256)
    ap.add_argument("--val-sweeps", type=int, default=4)
    ap.add_argument("--warm", action="store_true",
                    help="warm-start the regret net (skip fresh-init each iter)")
    ap.add_argument("--exact-cum", action="store_true",
                    help="regress regret net to EXACT RM+ cumulative regret "
                         "(FA-adequacy probe; no sampled buffer)")
    ap.add_argument("--neural-cum", action="store_true",
                    help="SCALABLE: store RM+ cumulative regret in the net "
                         "(warm incremental targets; no per-infoset table)")
    ap.add_argument("--ncum-gamma", type=float, default=0.99,
                    help="discount on the neural cumulative regret (DCFR-style)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; resumes if it exists (run more "
                         "iters across the per-call time cap)")
    a = ap.parse_args()
    if a.game == "kuhn":
        g, n = Kuhn, 3
    elif a.game == "kuhn6":
        g, n = make_kuhn_n(6), 6
    elif a.game == "leduc":
        g, n = Leduc, 3
    else:
        raise SystemExit("game must be kuhn|kuhn6|leduc")
    esc = NeuralESCHER(g, n, values=a.mode, hidden=a.hidden,
                       reg_steps=a.reg_steps, val_hidden=a.val_hidden, seed=0)
    if a.mode == "net":
        esc.val_sweeps = a.val_sweeps
    esc.warm = a.warm
    esc.exact_cum = a.exact_cum
    esc.neural_cum = a.neural_cum
    esc.ncum_gamma = a.ncum_gamma
    if a.ckpt and os.path.exists(a.ckpt):
        esc.load(a.ckpt)
        print(f"[resume] {a.game} (values={a.mode}) from iter {esc.t}, "
              f"+{a.iters} more:", flush=True)
    else:
        print(f"Neural ESCHER on {a.game} (values={a.mode}, hidden={a.hidden}, "
              f"reg_steps={a.reg_steps}):", flush=True)
    esc.run(iters=a.iters, eval_every=a.eval_every, ckpt=a.ckpt)


if __name__ == "__main__":
    main()
