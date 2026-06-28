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
        # full history feat (value net): both cards + public + hist
        self.full_dim = 2 * n_ranks + self.pub_dim + self.n_hist

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
        v = torch.zeros(self.full_dim)
        i = 0
        v[i + cards[0]] = 1.0; i += self.n_ranks
        v[i + cards[1]] = 1.0; i += self.n_ranks
        if self.has_public:
            v[i + self._public_idx(history, cards)] = 1.0; i += self.pub_dim
        v[i + self.hidx[history]] = 1.0
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
                 hidden=64, buf_cap=200_000, seed=0):
        self.g = game
        self.feat = Featurizer(game, n_ranks)
        self.values_mode = values
        self.A = game.n_actions
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.regret_net = MLP(self.feat.inf_dim, self.A, hidden).to(DEV)
        self.avg_net = MLP(self.feat.inf_dim, self.A, hidden).to(DEV)
        if values == "net":
            self.value_net = MLP(self.feat.full_dim, 1, hidden).to(DEV)
            self.value_opt = torch.optim.Adam(self.value_net.parameters(), 1e-3)
        self.reg_buf = Reservoir(buf_cap, seed + 1)
        self.strat_buf = Reservoir(buf_cap, seed + 2)

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

    @torch.no_grad()
    def net_value(self, history, cards):
        if self.g.is_terminal(history):
            return self.g.terminal_util_p0(history, cards)
        f = self.feat.full(history, cards).to(DEV)
        return self.value_net(f.unsqueeze(0)).item()

    def value_of(self, history, cards):
        return (self.net_value if self.values_mode == "net"
                else self.exact_value)(history, cards)

    # ── train the value net IS-free: bootstrap visited histories toward the
    #    on-σ expected child value, sampling trajectories with a fixed policy ──
    def train_value(self, n_traj=64, steps=8):
        g = self.g
        feats, targets = [], []
        for _ in range(n_traj):
            cards = self._sample_deal()
            path = []
            h = ""
            while not g.is_terminal(h):
                path.append((h, cards))
                legal = g.legal_actions(h)
                # fixed uniform sampling policy (the IS-free coverage policy)
                h = g.step_history(h, self.rng.choice(legal))
            for hist, c in path:
                legal = g.legal_actions(hist)
                player = g.current_player(hist)
                sig = self.sigma(hist, c, player, legal)
                tgt = 0.0
                for a in legal:
                    nxt = g.step_history(hist, a)
                    tgt += sig[a] * self.net_value(nxt, c)
                feats.append(self.feat.full(hist, c))
                targets.append(tgt)
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

        def rec(history, cards, p0, p1):
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
                                p1 * (sig[a] if player == 1 else 1.0))
                node += sig[a] * util_a[a]
            cf = p1 if player == 0 else p0
            own = p0 if player == 0 else p1
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

        for cards, _ in g.deals():
            rec("", cards, 1.0, 1.0)

        for I, rvec in inst.items():
            history, cards, player, legal = key_meta[I]
            f = self.feat.infoset(history, cards, player)
            self.reg_buf.add((f, torch.tensor(rvec), float(t)))
        for I, svec in strat.items():
            history, cards, player, legal = key_meta[I]
            tot = sum(svec)
            if tot > 0:
                f = self.feat.infoset(history, cards, player)
                self.strat_buf.add(
                    (f, torch.tensor([x / tot for x in svec]), float(t)))

    def _train_net(self, net, buf, steps, lr, mb=512):
        opt = torch.optim.Adam(net.parameters(), lr)
        data = buf.data
        if not data:
            return
        for _ in range(steps):
            idx = [self.rng.randrange(len(data)) for _ in range(min(mb, len(data)))]
            X = torch.stack([data[i][0] for i in idx]).to(DEV)
            Y = torch.stack([data[i][1] for i in idx]).to(DEV)
            W = torch.tensor([data[i][2] for i in idx]).to(DEV).unsqueeze(1)
            opt.zero_grad()
            pred = net(X)
            loss = (W * (pred - Y) ** 2).mean()
            loss.backward()
            opt.step()

    def fit_regret(self, steps=400, lr=3e-3):
        # fresh init each CFR iteration (Deep CFR): avoids stale-target drift.
        self.regret_net = MLP(self.feat.inf_dim, self.A).to(DEV)
        self._train_net(self.regret_net, self.reg_buf, steps, lr)

    def fit_avg(self, steps=600, lr=2e-3):
        self.avg_net = MLP(self.feat.inf_dim, self.A).to(DEV)
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

    def run(self, iters, eval_every=10):
        for t in range(1, iters + 1):
            if self.values_mode == "net":
                self.train_value(n_traj=64, steps=8)
            self.collect_regret(t)
            self.fit_regret()
            if t % eval_every == 0 or t == iters:
                self.fit_avg()
                expl = exploitability(self.g, self.avg_strategy())
                print(f"  iter {t:4d}   exploitability={expl:.5f}", flush=True)
        return expl


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "kuhn6"
    mode = sys.argv[2] if len(sys.argv) > 2 else "exact"
    if which == "kuhn":
        g, n = Kuhn, 3
    elif which == "kuhn6":
        g, n = make_kuhn_n(6), 6
    elif which == "leduc":
        g, n = Leduc, 3
    else:
        raise SystemExit("game must be kuhn|kuhn6|leduc")
    print(f"Neural ESCHER on {which} (values={mode}):")
    esc = NeuralESCHER(g, n, values=mode, seed=0)
    esc.run(iters=120, eval_every=10)


if __name__ == "__main__":
    main()
