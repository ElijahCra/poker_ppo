// HUNL range-vs-range terminal kernels for the ReBeL stage-2 subgame solver.
//
// Ranges are weight vectors over the 1326 hole-card combos (deck.h card ids
// 0..51, combo = unordered pair). Card removal is EXACT throughout via the
// standard inclusion-exclusion identities:
//   compatible-mass(i=(a,b)) = S − S_a − S_b + w_i
// (combos sharing one card subtracted once each; the identical combo — the
// only one sharing both cards — subtracted twice, added back once; holding
// the identical combo is impossible so its own weight never contributes to
// a payoff, only to the correction term.)
//
// Showdown uses Ray Wotton's TwoPlusTwo evaluator through
// Game::Utility::LookupHandValue (deck ids converted per CardConversion),
// with the classic sort-by-rank sweep: ascending rank groups maintain
// running total & per-card lower masses, tie masses come from the group
// itself. O(n log n) after the 1326 rank lookups.
//
// Every fast kernel has an O(n²) brute-force reference; `kernels` mode in
// hunl_main runs randomized equivalence tests — the stage-2 analogue of the
// Leduc exact-judge ladder.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace rebel_hunl {

inline constexpr int kCards = 52;
inline constexpr int kCombos = 1326;

struct ComboTable {
    std::array<std::array<uint8_t, 2>, kCombos> cards;  // combo -> (a < b)
    std::array<std::array<int16_t, kCards>, kCards> id; // (a,b) -> combo
    ComboTable();
    static const ComboTable& get();
};

// valid[i] = combo i shares no card with `board` (deck ids, `nb` cards).
void board_valid(const uint8_t* board, int nb, std::vector<uint8_t>& valid);

// mass[i] = Σ_j opp[j] over valid j sharing no card with i (0 for invalid i).
void compat_mass(const std::vector<double>& opp,
                 const std::vector<uint8_t>& valid, std::vector<double>& mass);

// Fold terminal: cfv[i] = u * compatible-mass(i). u = hero's chip payoff
// (constant across combos at a fold node).
void fold_cfv(const std::vector<double>& opp, const std::vector<uint8_t>& valid,
              double u, std::vector<double>& cfv);

// River showdown on a complete 5-card board:
//   cfv[i] = half_pot * (win_mass(i) − lose_mass(i)),  exact card removal.
void showdown_cfv(const std::vector<double>& opp, const uint8_t* board5,
                  double half_pot, std::vector<double>& cfv);

// Board-cached river evaluator: ranks, sort order and tie groups are fixed
// per board, so build once per solve and run the O(n) sweep per call.
// (showdown_cfv re-ranks and re-sorts every call — fine for tests, ruinous
// inside a 200-iteration solve.)
class RiverEval {
public:
    explicit RiverEval(const uint8_t* board5);
    void cfv(const std::vector<double>& opp, double half_pot,
             std::vector<double>& out) const;

private:
    std::vector<uint8_t> valid_;
    std::vector<int> order_;       // valid combos, rank-ascending
    std::vector<int> group_end_;   // exclusive end index per tie group
};

// O(n²) references for the randomized equivalence tests.
void compat_mass_brute(const std::vector<double>& opp,
                       const std::vector<uint8_t>& valid,
                       std::vector<double>& mass);
void showdown_cfv_brute(const std::vector<double>& opp, const uint8_t* board5,
                        double half_pot, std::vector<double>& cfv);

// 7-card rank of combo i on board5 (larger = stronger), -1 if i overlaps.
int combo_rank(int combo, const uint8_t* board5);

}  // namespace rebel_hunl
