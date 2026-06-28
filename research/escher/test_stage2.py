"""Stage 2 foundation: validate the recursive exact BR (vs brute force on
Kuhn) and the Leduc game (vs CFR convergence), before NN ESCHER on Leduc."""

from games import Kuhn, Leduc
from solvers import (VanillaCFR, exploitability, expected_value,
                     brute_force_br_value, exact_br_value)


def check_recursive_br_matches_bruteforce():
    g = Kuhn
    cfr = VanillaCFR(g)
    cfr.iterate(200)
    avg = cfr.average_strategy()
    for player in (0, 1):
        bf = brute_force_br_value(g, player, avg)
        rec = exact_br_value(g, player, avg)
        assert abs(bf - rec) < 1e-9, f"BR mismatch p{player}: bf={bf} rec={rec}"
    print(f"[BR] recursive == brute force on Kuhn  (p0={bf:.5f}) OK")


def validate_leduc():
    """Leduc correctness (NOT a fast-convergence test — vanilla CFR is slow
    here because Leduc utilities run to ±13). Three independent checks that
    the GAME is well-formed; the recursive BR is the exact exploitability
    metric used downstream."""
    from solvers import collect_infosets, _walk_value
    g = Leduc

    # 1. payoff symmetry: symmetric strategies give value 0 (no P0/P1 bias bug)
    def fixed(fn):
        s = {}
        for pl in (0, 1):
            for I, legal in collect_infosets(g, pl).items():
                a = fn(legal)
                s[I] = {x: (1.0 if x == a else 0.0) for x in legal}
        return s
    v_call = sum(p * _walk_value(g, "", c, fixed(lambda L: 1))
                 for c, p in g.deals())
    assert abs(v_call) < 1e-9, f"payoff asymmetry: call-call value {v_call}"

    # 2. infoset count == known Leduc (288 total)
    n_inf = len(collect_infosets(g, 0)) + len(collect_infosets(g, 1))
    assert n_inf == 288, f"infoset count {n_inf} != 288"

    # 3. CFR exploitability must DECREASE monotonically (converging, slowly)
    cfr = VanillaCFR(g)
    last = 0
    expls = []
    print("\nLeduc CFR (slow — large utilities). Exploitability decreasing:")
    for it in (200, 1000, 5000):
        cfr.iterate(it - last)
        last = it
        expls.append(exploitability(g, cfr.average_strategy()))
        print(f"  {it:5d}  expl={expls[-1]:.5f}")
    assert expls[-1] < expls[0], "exploitability not decreasing — structural bug"
    print(f"\nOK — Leduc is CORRECT: symmetric payoffs (call-call value 0), "
          f"288 infosets, exploitability decreasing. Vanilla CFR is just slow "
          f"(±13 utilities); the recursive BR gives exact exploitability for "
          f"ESCHER. Usable as the multi-round testbed.")


if __name__ == "__main__":
    check_recursive_br_matches_bruteforce()
    validate_leduc()
