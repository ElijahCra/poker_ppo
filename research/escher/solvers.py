"""
Exact-game tooling for the ESCHER prototype:
  - expected_value:   exact tree EV (P0 perspective) for two tabular strategies
  - collect_infosets: every infoset key + legal actions for a player
  - exploitability:   exact, via best response to the average strategy
  - VanillaCFR:       tabular CFR — the SANITY check that the testbed is right
                      (its exploitability must drive to ~0)

Best response is computed by BRUTE FORCE on Kuhn (enumerate the best-responder's
pure strategies, take the max-EV one) — obviously correct, zero algorithm risk,
which is the point of Stage 0. Larger games get the recursive BR later.
"""

from __future__ import annotations

import itertools
from collections import defaultdict


def _walk_value(game, history, cards, strat):
    """Exact P0-perspective EV of `history` under tabular `strat`
    (dict: infoset_key -> action-prob dict)."""
    if game.is_terminal(history):
        return game.terminal_util_p0(history, cards)
    I = game.infoset_key(history, cards)
    probs = strat[I]
    v = 0.0
    for a in game.legal_actions(history):
        pa = probs.get(a, 0.0)
        if pa == 0.0:
            continue
        v += pa * _walk_value(game, game.step_history(history, a), cards, strat)
    return v


def expected_value(game, strat) -> float:
    """Exact game value to P0, averaging over the deal."""
    return sum(p * _walk_value(game, "", cards, strat) for cards, p in game.deals())


def collect_infosets(game, player):
    """{infoset_key: sorted legal actions} for `player`, over the whole tree."""
    out = {}

    def rec(history, cards):
        if game.is_terminal(history):
            return
        if game.current_player(history) == player:
            I = game.infoset_key(history, cards)
            out.setdefault(I, list(game.legal_actions(history)))
        for a in game.legal_actions(history):
            rec(game.step_history(history, a), cards)

    seen_cards = set()
    for cards, _ in game.deals():
        if cards in seen_cards:
            continue
        seen_cards.add(cards)
        rec("", cards)
    return out


def _pure_strategies(infosets):
    """Yield every deterministic strategy (infoset -> single action)."""
    keys = list(infosets.keys())
    action_lists = [infosets[k] for k in keys]
    for combo in itertools.product(*action_lists):
        yield {k: {a: 1.0} for k, a in zip(keys, combo)}


def brute_force_br_value(game, br_player, opp_strat):
    """Best-responder's exact value vs fixed `opp_strat` (br_player view)."""
    br_infosets = collect_infosets(game, br_player)
    sign = 1.0 if br_player == 0 else -1.0
    best = -1e18
    for pure in _pure_strategies(br_infosets):
        merged = dict(opp_strat)
        merged.update(pure)
        v0 = expected_value(game, merged)
        v = sign * v0
        if v > best:
            best = v
    return best


def exact_br_value(game, br_player, sigma):
    """Best-responder's exact value vs fixed `sigma`, by backward induction
    over infosets (deepest first) — scales to Leduc where brute force can't.
    Validated to match brute_force_br_value on Kuhn."""
    from collections import defaultdict

    # Forward pass: each br infoset -> [(history, cards, opp+chance reach)].
    members = defaultdict(list)

    def fwd(history, cards, opp_reach):
        if game.is_terminal(history):
            return
        p = game.current_player(history)
        I = game.infoset_key(history, cards)
        legal = game.legal_actions(history)
        if p == br_player:
            members[I].append((history, cards, opp_reach))
            for a in legal:
                fwd(game.step_history(history, a), cards, opp_reach)
        else:
            sig = sigma[I]
            for a in legal:
                fwd(game.step_history(history, a), cards,
                    opp_reach * sig.get(a, 0.0))

    for cards, p in game.deals():
        fwd("", cards, p)

    br_action = {}

    def value(history, cards):
        if game.is_terminal(history):
            u = game.terminal_util_p0(history, cards)
            return u if br_player == 0 else -u
        p = game.current_player(history)
        I = game.infoset_key(history, cards)
        legal = game.legal_actions(history)
        if p == br_player:
            a = br_action[I]  # set: deeper than any infoset still being solved
            return value(game.step_history(history, a), cards)
        sig = sigma[I]
        return sum(sig.get(a, 0.0) * value(game.step_history(history, a), cards)
                   for a in legal)

    # Backward induction: solve deepest infosets first (a br node's children
    # are strictly deeper, so their br_action is already fixed).
    for I in sorted(members, key=lambda k: -len(members[k][0][0])):
        legal = game.legal_actions(members[I][0][0])
        q = {a: 0.0 for a in legal}
        for history, cards, reach in members[I]:
            for a in legal:
                q[a] += reach * value(game.step_history(history, a), cards)
        br_action[I] = max(legal, key=lambda a: q[a])

    return sum(p * value("", cards) for cards, p in game.deals())


def exploitability(game, avg_strat, *, exact=None) -> float:
    """Sum of both players' best-response values vs the average strategy.
    = 0 at a Nash equilibrium. Uses recursive exact BR for Leduc-sized games,
    brute force for Kuhn (auto by game name unless `exact` is forced)."""
    use_exact = (game.name != "kuhn") if exact is None else exact
    br = exact_br_value if use_exact else brute_force_br_value
    return br(game, 0, avg_strat) + br(game, 1, avg_strat)


# ─── Vanilla tabular CFR (sanity check) ─────────────────────────────────────

class VanillaCFR:
    def __init__(self, game):
        self.game = game
        self.regret = defaultdict(lambda: defaultdict(float))     # I -> a -> R
        self.strat_sum = defaultdict(lambda: defaultdict(float))  # I -> a -> Σσ
        self.infoset_actions = {}

    def _strategy(self, I, legal):
        r = self.regret[I]
        pos = {a: max(r[a], 0.0) for a in legal}
        s = sum(pos.values())
        if s > 0:
            return {a: pos[a] / s for a in legal}
        return {a: 1.0 / len(legal) for a in legal}

    def _cfr(self, history, cards, p0, p1):
        g = self.game
        if g.is_terminal(history):
            return g.terminal_util_p0(history, cards)
        player = g.current_player(history)
        legal = g.legal_actions(history)
        I = g.infoset_key(history, cards)
        self.infoset_actions[I] = legal
        sigma = self._strategy(I, legal)

        util_a = {}
        node_util = 0.0
        for a in legal:
            nxt = g.step_history(history, a)
            if player == 0:
                util_a[a] = self._cfr(nxt, cards, p0 * sigma[a], p1)
            else:
                util_a[a] = self._cfr(nxt, cards, p0, p1 * sigma[a])
            node_util += sigma[a] * util_a[a]

        # regret update for the acting player (P0-perspective utils → sign flip)
        cf_reach = p1 if player == 0 else p0
        own_reach = p0 if player == 0 else p1
        sign = 1.0 if player == 0 else -1.0
        for a in legal:
            regret = sign * (util_a[a] - node_util) * cf_reach
            self.regret[I][a] += regret
            self.strat_sum[I][a] += own_reach * sigma[a]
        return node_util

    def iterate(self, n_iters):
        for _ in range(n_iters):
            for cards, _ in self.game.deals():
                self._cfr("", cards, 1.0, 1.0)

    def average_strategy(self):
        avg = {}
        for I, ssum in self.strat_sum.items():
            legal = self.infoset_actions[I]
            tot = sum(ssum[a] for a in legal)
            if tot > 0:
                avg[I] = {a: ssum[a] / tot for a in legal}
            else:
                avg[I] = {a: 1.0 / len(legal) for a in legal}
        return avg
