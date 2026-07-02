"""
Stage 3b.0 — PAPER-EXACT ESCHER on Leduc (McAleer et al. 2022, arXiv 2206.04122),
validated against exact recursive BR (solvers.exact_br_value). This is the
RESEARCHED method, distinct from the escher_nn.py variant.

Faithful to the paper:
  - FIXED sampling policy for regret rollouts: the traverser samples ~ uniform
    b_i, the opponent ~ current π_{-i}. Outcome sampling (one line, no branch).
    No importance sampling, no explicit reach weight — the fixed sampler's
    per-infoset reach is constant across iterations and cancels under RM.
  - History value net Q_θ(h) → [A]: PRIVILEGED full history (both cards; public
    hidden until round 2 = history-faithful, so MC averages over the chance
    draw). Trained with MONTE-CARLO targets (terminal utility) from rollouts
    under the CURRENT π. NO bootstrapping (ESCHER's whole point vs DREAM).
  - Regret estimator  r̂(s,a) = q_i(h,a) − Σ_a' π_i(s,a') q_i(h,a').
  - Deep-CFR regret nets R_0, R_1: reservoir buffer (accumulates across iters →
    cumulative regret via iteration weighting), REINIT each iteration, RM⁺.
  - Average policy net π̄_φ: classification on (infoset, a_taken) sampled under
    the current π. This is the DEPLOYED, exploitability-judged strategy.

Run in WSL: python3 escher_paper.py [leduc|kuhn6|kuhn] [iters]
"""

from __future__ import annotations

import sys
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from games import Kuhn, Leduc, make_kuhn_n
from solvers import exploitability

DEV = "cpu"


# ─── features (history-faithful: public hidden until revealed) ───────────────
class Featurizer:
    def __init__(self, game, n_ranks):
        self.g = game
        self.n_ranks = n_ranks
        keys, self.has_public = self._enumerate()
        hists = sorted({k.split("|")[-1] for k in keys})
        self.hidx = {h: i for i, h in enumerate(hists)}
        self.n_hist = len(hists)
        self.pub_dim = (n_ranks + 1) if self.has_public else 0  # 0=hidden,1+rank
        self.inf_dim = 1 + n_ranks + self.pub_dim + self.n_hist + 1
        self.full_dim = 2 * n_ranks + self.pub_dim + self.n_hist

    def _enumerate(self):
        g, keys, seen, pub = self.g, set(), set(), False

        def rec(h, cards):
            if g.is_terminal(h):
                return
            keys.add(g.infoset_key(h, cards))
            for a in g.legal_actions(h):
                rec(g.step_history(h, a), cards)

        for cards, _ in g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        pub = any("/" in k.split("|")[-1] for k in keys)
        return keys, pub

    def _pub_idx(self, h, cards):
        if not self.has_public:
            return 0
        return (1 + cards[2]) if "/" in h else 0  # hidden until round 2

    def infoset(self, h, cards, player):
        v = torch.zeros(self.inf_dim)
        i = 0
        card = cards[player]
        v[i] = card / max(1, self.n_ranks - 1); i += 1
        v[i + card] = 1.0; i += self.n_ranks
        if self.has_public:
            v[i + self._pub_idx(h, cards)] = 1.0; i += self.pub_dim
        v[i + self.hidx[h]] = 1.0; i += self.n_hist
        v[i] = float(player)
        return v

    def full(self, h, cards):
        v = torch.zeros(self.full_dim)
        i = 0
        v[i + cards[0]] = 1.0; i += self.n_ranks
        v[i + cards[1]] = 1.0; i += self.n_ranks
        if self.has_public:
            v[i + self._pub_idx(h, cards)] = 1.0; i += self.pub_dim
        v[i + self.hidx[h]] = 1.0
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
        self.cap, self.data, self.n = cap, [], 0
        self.rng = random.Random(seed)

    def add(self, item):
        self.n += 1
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            j = self.rng.randint(0, self.n - 1)
            if j < self.cap:
                self.data[j] = item


# ─── paper-exact ESCHER ──────────────────────────────────────────────────────
class ESCHERPaper:
    def __init__(self, game, n_ranks, hidden=64, val_hidden=128,
                 buf_cap=400_000, seed=0):
        self.g = game
        self.feat = Featurizer(game, n_ranks)
        self.A = game.n_actions
        self.hidden, self.val_hidden = hidden, val_hidden
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.value_net = MLP(self.feat.full_dim, self.A, val_hidden).to(DEV)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), 1e-3)
        self.regret_net = [None, None]          # reinit each iteration
        self.avg_net = MLP(self.feat.inf_dim, self.A, hidden).to(DEV)
        self.reg_buf = [Reservoir(buf_cap, seed + 1), Reservoir(buf_cap, seed + 2)]
        self.val_buf = Reservoir(buf_cap, seed + 3)
        self.avg_buf = Reservoir(buf_cap, seed + 4)
        # tabular average diagnostic (the ceiling the avg net could reach)
        self.tab_ss = defaultdict(lambda: [0.0] * self.A)
        self.tab_w = defaultdict(float)
        self.tab_legal = {}
        self.val_traj = 2400   # current-π MC rollouts/iter (value net is data-hungry)
        self.val_steps = 600
        self.reg_steps = 400   # Deep-CFR regret-fit budget (paper: ~5000 @ 2048)
        self.reg_mb = 512
        self.rm_plus = False   # paper uses plain RM; RM⁺ is a researched lever
        # action-conditioned Q only trains the taken action per sample; under
        # pure π, rarely-played actions never get a target. Mix in exploration
        # so every action gets MC coverage (slight bias toward V^{π_ε}).
        self.val_expl = 0.5

    # current strategy at an infoset: RM⁺ on the regret net (uniform if unset)
    @torch.no_grad()
    def sigma(self, h, cards, player, legal):
        net = self.regret_net[player]
        if net is None:
            return {a: 1.0 / len(legal) for a in legal}
        r = net(self.feat.infoset(h, cards, player).unsqueeze(0)).squeeze(0)
        pos = torch.clamp(r, min=0.0)
        s = sum(pos[a].item() for a in legal)
        if s > 1e-12:
            return {a: pos[a].item() / s for a in legal}
        return {a: 1.0 / len(legal) for a in legal}

    @torch.no_grad()
    def q_p0(self, h, cards):
        """Q(h,·) in P0 perspective. Net by default; exact on-σ child values
        when use_exact_value (isolates the pipeline from the MC value net)."""
        if getattr(self, "use_exact_value", False):
            q = torch.zeros(self.A)
            for a in self.g.legal_actions(h):
                q[a] = self._exact_child(self.g.step_history(h, a), cards)
            return q
        return self.value_net(self.feat.full(h, cards).unsqueeze(0)).squeeze(0)

    def _exact_child(self, h, cards):
        g = self.g
        if g.is_terminal(h):
            return g.terminal_util_p0(h, cards)
        p = g.current_player(h)
        legal = g.legal_actions(h)
        sig = self.sigma(h, cards, p, legal)
        return sum(sig[a] * self._exact_child(g.step_history(h, a), cards)
                   for a in legal)

    def _sample(self, legal, sig):
        r, acc = self.rng.random(), 0.0
        for a in legal:
            acc += sig[a]
            if r <= acc:
                return a
        return legal[-1]

    def _sample_deal(self):
        deals = self.g.deals()
        r, acc = self.rng.random(), 0.0
        for cards, p in deals:
            acc += p
            if r <= acc:
                return cards
        return deals[-1][0]

    # ── value data: rollouts under the CURRENT π (both players), MC target ──
    # The buffer is CLEARED each iteration: the regret needs v^{σ_t} (the value
    # under the CURRENT strategy), so Q must be trained on current-π data, not a
    # reservoir blending all past policies.
    def collect_value(self, n_traj):
        g = self.g
        self.val_buf.data = []
        self.val_buf.n = 0
        for _ in range(n_traj):
            cards = self._sample_deal()
            path, h = [], ""
            while not g.is_terminal(h):
                p = g.current_player(h)
                legal = g.legal_actions(h)
                if self.rng.random() < self.val_expl:
                    a = legal[self.rng.randrange(len(legal))]
                else:
                    a = self._sample(legal, self.sigma(h, cards, p, legal))
                path.append((self.feat.full(h, cards), a))
                h = g.step_history(h, a)
            z = g.terminal_util_p0(h, cards)  # P0 perspective
            for ffeat, a in path:
                self.val_buf.add((ffeat, a, z))

    def train_value(self, steps=400, mb=512):
        data = self.val_buf.data
        if not data:
            return
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(mb, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            acts = torch.tensor([data[i][1] for i in idx])
            y = torch.tensor([data[i][2] for i in idx], dtype=torch.float32)
            self.value_opt.zero_grad()
            q = self.value_net(X)
            pred = q.gather(1, acts.unsqueeze(1)).squeeze(1)  # Q(h, a_taken)
            loss = F.mse_loss(pred, y)
            loss.backward()
            self.value_opt.step()

    # ── regret data: FIXED sampler (traverser ~ uniform, opp ~ π), value-net
    #    action values, no reach weight. ──
    def collect_regret(self, player, t, n_traj):
        g = self.g
        sign = 1.0 if player == 0 else -1.0
        for _ in range(n_traj):
            cards = self._sample_deal()
            h = ""
            while not g.is_terminal(h):
                p = g.current_player(h)
                legal = g.legal_actions(h)
                if p == player:
                    qi = sign * self.q_p0(h, cards)           # player-i view
                    sig = self.sigma(h, cards, p, legal)
                    vbar = sum(sig[a] * qi[a].item() for a in legal)
                    rvec = [0.0] * self.A
                    for a in legal:
                        rvec[a] = qi[a].item() - vbar
                    self.reg_buf[player].add(
                        (self.feat.infoset(h, cards, p), torch.tensor(rvec),
                         float(t)))
                    a = legal[self.rng.randrange(len(legal))]  # fixed b_i: uniform
                else:
                    a = self._sample(legal, self.sigma(h, cards, p, legal))
                h = g.step_history(h, a)

    def fit_regret(self, player, lr=3e-3):
        # Deep CFR: REINIT, fit the reservoir (linear-weighted = cumulative),
        # global-normalised (RM is scale-invariant).
        steps, mb = self.reg_steps, self.reg_mb
        self.regret_net[player] = MLP(self.feat.inf_dim, self.A, self.hidden).to(DEV)
        data = self.reg_buf[player].data
        if not data:
            return
        opt = torch.optim.Adam(self.regret_net[player].parameters(), lr)
        ss = sum(float(x) ** 2 for d in data for x in d[1])
        scale = max(1e-6, (ss / max(1, len(data) * self.A)) ** 0.5)
        net = self.regret_net[player]
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(mb, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            Y = torch.stack([data[i][1] for i in idx]).to(DEV) / scale
            W = torch.tensor([data[i][2] for i in idx]).to(DEV).unsqueeze(1)
            opt.zero_grad()
            loss = (W * (net(X) - Y) ** 2).mean()
            loss.backward()
            opt.step()

    # ── average data: rollouts under the CURRENT π, store (infoset, a_taken) ──
    def collect_avg(self, t, n_traj):
        g = self.g
        for _ in range(n_traj):
            cards = self._sample_deal()
            h = ""
            while not g.is_terminal(h):
                p = g.current_player(h)
                legal = g.legal_actions(h)
                sig = self.sigma(h, cards, p, legal)
                a = self._sample(legal, sig)
                I = g.infoset_key(h, cards)
                self.avg_buf.add((self.feat.infoset(h, cards, p), a, float(t)))
                self.tab_ss[I][a] += t
                self.tab_w[I] += t
                self.tab_legal[I] = legal
                h = g.step_history(h, a)

    def train_avg(self, steps=600, lr=2e-3, mb=1024):
        data = self.avg_buf.data
        if not data:
            return
        self.avg_net = MLP(self.feat.inf_dim, self.A, self.hidden).to(DEV)
        opt = torch.optim.Adam(self.avg_net.parameters(), lr)
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(mb, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            acts = torch.tensor([data[i][1] for i in idx])
            W = torch.tensor([data[i][2] for i in idx]).to(DEV)
            opt.zero_grad()
            logp = F.log_softmax(self.avg_net(X), dim=1)
            loss = (W * F.nll_loss(logp, acts, reduction="none")).mean()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def avg_strategy(self):
        out, seen, g = {}, set(), self.g

        def rec(h, cards):
            if g.is_terminal(h):
                return
            I = g.infoset_key(h, cards)
            legal = g.legal_actions(h)
            if I not in out:
                p = g.current_player(h)
                logits = self.avg_net(self.feat.infoset(h, cards, p).unsqueeze(0)).squeeze(0)
                m = torch.full((self.A,), float("-inf"))
                for a in legal:
                    m[a] = logits[a]
                pr = F.softmax(m, dim=0).tolist()
                out[I] = {a: pr[a] for a in legal}
            for a in legal:
                rec(g.step_history(h, a), cards)

        for cards, _ in g.deals():
            if cards in seen:
                continue
            seen.add(cards)
            rec("", cards)
        return out

    def tabular_average(self):
        out, seen, g = {}, set(), self.g

        def rec(h, cards):
            if g.is_terminal(h):
                return
            I = g.infoset_key(h, cards)
            legal = g.legal_actions(h)
            if I not in out:
                w = self.tab_w.get(I, 0.0)
                if w > 0:
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

    @torch.no_grad()
    def value_diag(self):
        """Reach-weighted RMSE of net Q(h,a) vs the EXACT on-σ value of child
        h·a (P0 perspective) — does the value net track v^{σ_t}?"""
        g = self.g
        acc = {"se": 0.0, "w": 0.0, "mx": 0.0}
        memo = {}

        def exact_val(h, cards):
            if g.is_terminal(h):
                return g.terminal_util_p0(h, cards)
            key = (h, cards)
            if key in memo:
                return memo[key]
            p = g.current_player(h)
            legal = g.legal_actions(h)
            sig = self.sigma(h, cards, p, legal)
            v = sum(sig[a] * exact_val(g.step_history(h, a), cards) for a in legal)
            memo[key] = v
            return v

        def rec(h, cards, reach):
            if g.is_terminal(h):
                return
            p = g.current_player(h)
            legal = g.legal_actions(h)
            sig = self.sigma(h, cards, p, legal)
            q = self.q_p0(h, cards)
            for a in legal:
                qe = exact_val(g.step_history(h, a), cards)
                e = q[a].item() - qe
                acc["se"] += reach * e * e
                acc["w"] += reach
                acc["mx"] = max(acc["mx"], abs(e))
                rec(g.step_history(h, a), cards, reach * sig[a])

        for cards, pp in g.deals():
            rec("", cards, pp)
        return (acc["se"] / max(acc["w"], 1e-9)) ** 0.5, acc["mx"]

    def run(self, iters, eval_every=10, n_traj=300):
        for t in range(1, iters + 1):
            if not getattr(self, "use_exact_value", False):
                self.collect_value(self.val_traj)  # current-π MC
                self.train_value(steps=self.val_steps)
            for player in (0, 1):
                self.collect_regret(player, t, n_traj)
                self.fit_regret(player)
            self.collect_avg(t, n_traj)
            if t % eval_every == 0 or t == iters:
                self.train_avg()
                expl_net = exploitability(self.g, self.avg_strategy())
                expl_tab = exploitability(self.g, self.tabular_average())
                extra = ""
                if not getattr(self, "use_exact_value", False):
                    vr, vmx = self.value_diag()
                    extra = f"   V-rmse={vr:.4f} (max {vmx:.3f})"
                print(f"  iter {t:4d}   expl(avg-net)={expl_net:.5f}   "
                      f"expl(tab-avg)={expl_tab:.5f}{extra}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "leduc"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    if which == "kuhn":
        g, n = Kuhn, 3
    elif which == "kuhn6":
        g, n = make_kuhn_n(6), 6
    elif which == "leduc":
        g, n = Leduc, 3
    else:
        raise SystemExit("game must be kuhn|kuhn6|leduc")
    exact = len(sys.argv) > 3 and sys.argv[3] == "exact"
    print(f"PAPER-EXACT ESCHER on {which}{' (exact values)' if exact else ''}:")
    esc = ESCHERPaper(g, n, seed=0)
    esc.use_exact_value = exact
    esc.run(iters, eval_every=10)


if __name__ == "__main__":
    main()
