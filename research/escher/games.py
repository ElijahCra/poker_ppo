"""
Tiny imperfect-information games with EXACT exploitability, for validating the
ESCHER prototype before any C++/HUNL port.

Both games expose the same minimal extensive-form interface the solvers need:
  - a recursive tree of histories (strings), with chance at the root (the deal),
  - infoset keys (what the acting player observes),
  - legal actions, terminal payoffs (player-0 perspective, zero-sum).

Kuhn poker is the canonical 12-infoset sanity game; Leduc is the larger
(deal + public card) stress test. Exact best-response / exploitability lives
in `solvers.py` and walks this same tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Optional


# ─── Kuhn poker ─────────────────────────────────────────────────────────────
# 3 cards {0=J,1=Q,2=K}, one each, 1-chip ante. P0 acts first.
# Actions: 'p' (pass/check) and 'b' (bet/call 1). Standard Kuhn betting tree.

class Kuhn:
    name = "kuhn"
    n_actions = 2
    ACTIONS = ["p", "b"]  # pass/check, bet/call

    @staticmethod
    def deals():
        # (p0_card, p1_card), uniform over the 6 orderings, prob 1/6 each.
        return [(c, 1.0 / 6.0) for c in permutations(range(3), 2)]

    @staticmethod
    def is_terminal(history: str) -> bool:
        return history in ("pp", "bp", "bb", "pbp", "pbb")

    @staticmethod
    def current_player(history: str) -> int:
        # alternates from P0; chance (the deal) handled by deals().
        return len(history) % 2

    @staticmethod
    def legal_actions(history: str):
        return [0, 1]  # both actions always legal in Kuhn

    @staticmethod
    def step_history(history: str, action: int) -> str:
        return history + Kuhn.ACTIONS[action]

    @staticmethod
    def infoset_key(history: str, cards) -> str:
        return f"{cards[Kuhn.current_player(history)]}|{history}"

    @staticmethod
    def terminal_util_p0(history: str, cards) -> float:
        # Returns player-0 net chips (zero-sum). Antes already in the pot.
        p0, p1 = cards
        p0_wins = 1 if p0 > p1 else -1  # higher card wins at showdown
        if history == "pp":           # both check: showdown for the antes (1)
            return p0_wins
        if history == "bp":           # P0 bet, P1 folded → P0 wins ante (1)
            return 1
        if history == "pbp":          # P0 check, P1 bet, P0 folded → P0 loses 1
            return -1
        if history in ("bb", "pbb"):  # both put a bet in: showdown for 2
            return 2 * p0_wins
        raise ValueError(f"non-terminal history {history!r}")


# ─── Leduc poker (simplified) ───────────────────────────────────────────────
# 6-card deck = two suits of {J,Q,K} (ranks 0,1,2). Each player 1 private card;
# one public card revealed after round 1. Ante 1. Bets: round-1 size 2, round-2
# size 4, max 2 raises per round. Pair-with-public beats high card; else higher
# card wins. (Standard simplified Leduc used across the CFR literature.)

class Leduc:
    name = "leduc"
    n_actions = 3
    ACTIONS = ["f", "c", "r"]  # fold, call/check, raise
    RANKS = 3
    DECK = [0, 0, 1, 1, 2, 2]   # two of each rank

    @staticmethod
    def deals():
        seen = {}
        out = []
        deck = Leduc.DECK
        n = len(deck)
        # p0, p1, public — distinct deck positions, collapse identical rank
        # combos by probability.
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    key = (deck[i], deck[j], deck[k])
                    out.append((key, 1.0))
        # normalise by count of identical (rank) outcomes
        total = len(out)
        agg = {}
        for key, _ in out:
            agg[key] = agg.get(key, 0.0) + 1.0 / total
        return list(agg.items())

    # History encodes both rounds, separated by '/'. A round string is a
    # sequence over {c,r,f}. Round ends when a call closes action (after a
    # raise) or both check.
    @staticmethod
    def _round_over(rnd: str) -> bool:
        if rnd.endswith("f"):
            return True
        if rnd == "cc":
            return True
        # a call after at least one raise closes the round
        if rnd.endswith("c") and "r" in rnd:
            return True
        return False

    @staticmethod
    def is_terminal(history: str) -> bool:
        if history.endswith("f"):
            return True
        rounds = history.split("/")
        if len(rounds) == 2 and Leduc._round_over(rounds[1]):
            return True
        return False

    @staticmethod
    def _cur_round(history: str) -> str:
        return history.split("/")[-1]

    @staticmethod
    def current_player(history: str) -> int:
        rnd = Leduc._cur_round(history)
        # P0 acts first each round (after the public card P0 acts first too).
        return len(rnd) % 2

    @staticmethod
    def n_raises(rnd: str) -> int:
        return rnd.count("r")

    @staticmethod
    def legal_actions(history: str):
        rnd = Leduc._cur_round(history)
        acts = [1]  # call/check always legal
        facing_bet = rnd.endswith("r")
        if facing_bet:
            acts.append(0)  # fold only when facing a bet/raise
        if Leduc.n_raises(rnd) < 2:
            acts.append(2)  # raise if under the cap
        return sorted(acts)

    @staticmethod
    def infoset_key(history: str, cards) -> str:
        p = Leduc.current_player(history)
        # acting player sees own card; public card only after round 1.
        public = cards[2] if "/" in history else "?"
        return f"{cards[p]}|{public}|{history}"

    @staticmethod
    def _pot_and_round_advance(history: str):
        # Returns (history_after_round_advance_or_same, just_finished_round1).
        rounds = history.split("/")
        if len(rounds) == 1 and Leduc._round_over(rounds[0]) and not history.endswith("f"):
            return history + "/", True
        return history, False

    @staticmethod
    def step_history(history: str, action: int) -> str:
        h = history + Leduc.ACTIONS[action]
        # advance to round 2 (reveal public) when round 1 closes by call/check
        rounds = h.split("/")
        if len(rounds) == 1 and Leduc._round_over(rounds[0]) and not h.endswith("f"):
            h = h + "/"
        return h

    @staticmethod
    def terminal_util_p0(history: str, cards) -> float:
        p0, p1, pub = cards
        # contributions per player (ante 1 + bets). Round-1 raise size 2,
        # round-2 size 4.
        contrib = [1, 1]
        rounds = history.split("/")
        for ri, rnd in enumerate(rounds):
            size = 2 if ri == 0 else 4
            # rebuild contributions by replaying the round
            cur = [0, 0]
            to_call = 0
            actor = 0
            for ch in rnd:
                if ch == "c":
                    cur[actor] += to_call
                elif ch == "r":
                    cur[actor] += to_call + size
                    to_call = size
                # fold adds nothing
                actor ^= 1
            contrib[0] += cur[0]
            contrib[1] += cur[1]
        if history.endswith("f"):
            folder = Leduc.current_player(history)  # the player to act folded? no:
            # the LAST action was 'f' by the player who just acted; that player
            # is the previous actor.
            # Recompute: the folding player is the one who played the final 'f'.
            # current_player(history) after appending 'f' would be the NEXT, so
            # the folder is 1 - that. Use parity of the round up to the fold.
            rnd = Leduc._cur_round(history)
            folder = (len(rnd) - 1) % 2
            winner = 1 - folder
            return (contrib[1 - winner]) if winner == 0 else -(contrib[1 - winner])
        # showdown
        def rank(card):
            return 10 + card if card == pub else card  # pair beats high card
        r0, r1 = rank(p0), rank(p1)
        if r0 == r1:
            return 0.0
        winner = 0 if r0 > r1 else 1
        # winner takes the loser's contribution (net), zero-sum
        return contrib[1] if winner == 0 else -contrib[0]


# ─── N-card Kuhn ────────────────────────────────────────────────────────────
# Same single-round check/bet tree as Kuhn but with N>3 ranks (one card each),
# higher card wins. More infosets (N · histories) so a NN must generalise over
# card value — a clean, trivially-CORRECT scaling testbed for Stage-2 NN ESCHER
# (Leduc, multi-round, is deferred pending a value-bug fix). Built by a factory
# so `name`/`deals` carry N.

def make_kuhn_n(n_cards: int):
    class KuhnN(Kuhn):
        name = f"kuhn{n_cards}"
        N = n_cards

        @staticmethod
        def deals():
            from itertools import permutations as _perm
            cs = list(_perm(range(n_cards), 2))
            p = 1.0 / len(cs)
            return [(c, p) for c in cs]

    return KuhnN


GAMES = {"kuhn": Kuhn, "leduc": Leduc}
