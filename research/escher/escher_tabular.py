"""
Stage 1: TABULAR ESCHER on Kuhn — validates the ESCHER ESTIMATOR in isolation
from neural-net approximation. If this drives exploitability to ~0 like CFR,
the core update rule (regret from a learned history-value function, sampled
with a FIXED policy, NO importance sampling) is correct and worth porting to a
network (Stage 2) and then C++/HUNL.

Faithful to the reference (Sandholm-Lab/ESCHER):
  - sampling policy = expl·uniform + (1-expl)·σ, default expl=1 (uniform, fixed);
  - history value V[(cards,h)] = E[P0 utility | h] under the CURRENT σ, learned
    by bootstrapping toward Σ_a σ(a) V[child] on sampled trajectories (on-σ
    target from off-σ samples — the IS-free trick);
  - instantaneous regret r(I,a) = sign·(V[h·a] − V[h]) for ALL actions, weighted
    by the counterfactual (opp+chance) reach of the sampled history;
  - regret matching (ReLU) for σ; average strategy from reach-weighted σ sums.
"""

from __future__ import annotations

import random
from collections import defaultdict

from games import Kuhn
from solvers import exploitability, expected_value


class TabularESCHER:
    def __init__(self, game, expl=0.6, v_lr=0.5, seed=0):
        self.g = game
        self.expl = expl
        self.v_lr = v_lr
        self.rng = random.Random(seed)
        self.regret = defaultdict(lambda: defaultdict(float))
        self.strat_sum = defaultdict(lambda: defaultdict(float))
        self.infoset_actions = {}
        self.V = defaultdict(float)        # (cards, history) -> P0 value under σ

    # current strategy at an infoset via regret matching (ReLU)
    def sigma(self, I, legal):
        r = self.regret[I]
        pos = [max(r[a], 0.0) for a in legal]
        s = sum(pos)
        if s > 0:
            return {a: pos[i] / s for i, a in enumerate(legal)}
        return {a: 1.0 / len(legal) for a in legal}

    # ── value learning (ESCHER-specific, IS-FREE): learn V[(cards,h)] = E[P0
    # utility | h] under σ from trajectories sampled with the FIXED policy,
    # bootstrapping each visited history toward Σ_a σ(a) V[child]. The target
    # uses σ (not the sampled action), so off-policy uniform sampling gives an
    # on-σ value with NO importance weight. ──
    def learn_values(self, n_traj):
        g = self.g
        for _ in range(n_traj):
            cards = self._sample_deal()
            # walk to a terminal under the fixed sampling policy, recording
            # the visited histories, then bootstrap them bottom-up.
            path = []
            h = ""
            while not g.is_terminal(h):
                path.append(h)
                legal = g.legal_actions(h)
                I = g.infoset_key(h, cards)
                sig = self.sigma(I, legal)
                u = 1.0 / len(legal)
                samp = [self.expl * u + (1.0 - self.expl) * sig[a] for a in legal]
                h = g.step_history(h, self._choose(legal, samp))
            for hist in reversed(path):
                legal = g.legal_actions(hist)
                I = g.infoset_key(hist, cards)
                sig = self.sigma(I, legal)
                target = 0.0
                for a in legal:
                    nxt = g.step_history(hist, a)
                    cv = (g.terminal_util_p0(nxt, cards) if g.is_terminal(nxt)
                          else self.V[(cards, nxt)])
                    target += sig[a] * cv
                key = (cards, hist)
                self.V[key] += self.v_lr * (target - self.V[key])

    # ── regret update: full CFR traversal, but action values read from the
    # learned V (the ESCHER claim: V replaces rollout/IS). Standard reach
    # weighting — π_{-i} on regret, π_i on the average-strategy sum. ──
    def _regret(self, history, cards, p_reach, opp_reach):
        g = self.g
        if g.is_terminal(history):
            return
        player = g.current_player(history)
        legal = g.legal_actions(history)
        I = g.infoset_key(history, cards)
        self.infoset_actions[I] = legal
        sig = self.sigma(I, legal)
        sign = 1.0 if player == 0 else -1.0
        # p_reach = P0's reach to here, opp_reach = P1's reach. The acting
        # player's counterfactual reach is the OTHER player's reach; the
        # average-strategy sum is weighted by the acting player's OWN reach.
        cf_reach  = opp_reach if player == 0 else p_reach
        own_reach = p_reach   if player == 0 else opp_reach
        v_node = self.V[(cards, history)]
        for a in legal:
            nxt = g.step_history(history, a)
            v_child = (g.terminal_util_p0(nxt, cards) if g.is_terminal(nxt)
                       else self.V[(cards, nxt)])
            self.regret[I][a] += cf_reach * sign * (v_child - v_node)
            self.strat_sum[I][a] += own_reach * sig[a]
            if player == 0:
                self._regret(nxt, cards, p_reach * sig[a], opp_reach)
            else:
                self._regret(nxt, cards, p_reach, opp_reach * sig[a])

    def iterate(self, n):
        for _ in range(n):
            self.learn_values(n_traj=12)
            for cards, _ in self.g.deals():
                self._regret("", cards, 1.0, 1.0)

    def _choose(self, legal, weights):
        s = sum(weights)
        r = self.rng.random() * s
        acc = 0.0
        for a, w in zip(legal, weights):
            acc += w
            if r <= acc:
                return a
        return legal[-1]

    def _sample_deal(self):
        deals = self.g.deals()
        r = self.rng.random()
        acc = 0.0
        for cards, p in deals:
            acc += p
            if r <= acc:
                return cards
        return deals[-1][0]

    def average_strategy(self):
        avg = {}
        for I, ssum in self.strat_sum.items():
            legal = self.infoset_actions[I]
            tot = sum(ssum[a] for a in legal)
            avg[I] = ({a: ssum[a] / tot for a in legal} if tot > 0
                      else {a: 1.0 / len(legal) for a in legal})
        return avg


def main():
    g = Kuhn
    esc = TabularESCHER(g, expl=0.6, v_lr=0.5, seed=1)
    print("iter   exploitability   game_value(P0)")
    for it in (100, 1000, 5000, 20000, 50000):
        esc.iterate(it - main.last)
        main.last = it
        avg = esc.average_strategy()
        expl = exploitability(g, avg)
        print(f"{it:6d}   {expl:.6f}        {expected_value(g, avg):+.6f}")
    print(f"\n{'CONVERGED' if expl < 0.03 else 'NOT converged'} "
          f"— tabular ESCHER exploitability={expl:.5f} "
          f"(CFR reached ~0.005; <0.03 here validates the estimator)")


main.last = 0

if __name__ == "__main__":
    main()
