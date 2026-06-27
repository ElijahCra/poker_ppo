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
    g = Leduc
    cfr = VanillaCFR(g)
    print("\nLeduc CFR (recursive-BR exploitability):")
    print("iter   exploitability   game_value(P0)")
    last = 0
    for it in (1, 10, 50, 200, 1000):
        cfr.iterate(it - last)
        last = it
        avg = cfr.average_strategy()
        expl = exploitability(g, avg)
        print(f"{it:5d}   {expl:.6f}        {expected_value(g, avg):+.6f}")
    assert expl < 0.05, f"Leduc CFR did not converge (expl {expl}) — game bug?"
    print(f"\nOK — Leduc solves (exploitability {expl:.4f}); the game + "
          f"recursive BR are correct. Ready for NN ESCHER.")


if __name__ == "__main__":
    check_recursive_br_matches_bruteforce()
    validate_leduc()
