"""Stage 0: validate the testbed on Kuhn.

Two checks that must both pass before any ESCHER code is trusted:
  1. Vanilla CFR's exploitability drives to ~0 (the solver + exploitability
     metric agree on a known-solvable game).
  2. The converged game value matches Kuhn's analytic value, -1/18 to P0.
"""

from games import Kuhn
from solvers import VanillaCFR, exploitability, expected_value


def main():
    g = Kuhn
    cfr = VanillaCFR(g)
    print("iter   exploitability   game_value(P0)")
    for it in (1, 10, 100, 1000, 5000):
        cfr.iterate(it if it == 1 else it - last)  # cumulative
        last = it
        avg = cfr.average_strategy()
        expl = exploitability(g, avg)
        gv = expected_value(g, avg)
        print(f"{it:5d}   {expl:.6f}        {gv:+.6f}")

    # Assertions: CFR must converge.
    assert expl < 1e-2, f"CFR did not converge (exploitability {expl})"
    assert abs(gv - (-1.0 / 18.0)) < 5e-3, f"game value {gv} != -1/18"
    print(f"\nOK — Kuhn solved. exploitability={expl:.5f}, "
          f"game value={gv:+.5f} (analytic -1/18={-1/18:+.5f})")


if __name__ == "__main__":
    last = 0
    main()
