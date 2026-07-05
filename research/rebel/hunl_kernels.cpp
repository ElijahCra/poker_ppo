#include "hunl_kernels.h"

#include <algorithm>
#include <numeric>

#include "Utility/CardConversion.hpp"
#include "Utility/Utility.hpp"

namespace rebel_hunl {

ComboTable::ComboTable() {
    int k = 0;
    for (auto& row : id) row.fill(-1);
    for (int a = 0; a < kCards; ++a)
        for (int b = a + 1; b < kCards; ++b) {
            cards[k] = {static_cast<uint8_t>(a), static_cast<uint8_t>(b)};
            id[a][b] = static_cast<int16_t>(k);
            id[b][a] = static_cast<int16_t>(k);
            ++k;
        }
}

const ComboTable& ComboTable::get() {
    static const ComboTable t;
    return t;
}

void board_valid(const uint8_t* board, int nb, std::vector<uint8_t>& valid) {
    const auto& ct = ComboTable::get();
    bool dead[kCards] = {};
    for (int i = 0; i < nb; ++i) dead[board[i]] = true;
    valid.assign(kCombos, 1);
    for (int i = 0; i < kCombos; ++i)
        if (dead[ct.cards[i][0]] || dead[ct.cards[i][1]]) valid[i] = 0;
}

void compat_mass(const std::vector<double>& opp,
                 const std::vector<uint8_t>& valid,
                 std::vector<double>& mass) {
    const auto& ct = ComboTable::get();
    double S = 0.0;
    double Sc[kCards] = {};
    for (int j = 0; j < kCombos; ++j) {
        if (!valid[j]) continue;
        const double w = opp[j];
        S += w;
        Sc[ct.cards[j][0]] += w;
        Sc[ct.cards[j][1]] += w;
    }
    mass.assign(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i) {
        if (!valid[i]) continue;
        const int a = ct.cards[i][0], b = ct.cards[i][1];
        mass[i] = S - Sc[a] - Sc[b] + opp[i];
    }
}

void fold_cfv(const std::vector<double>& opp, const std::vector<uint8_t>& valid,
              double u, std::vector<double>& cfv) {
    compat_mass(opp, valid, cfv);
    for (double& v : cfv) v *= u;
}

int combo_rank(int combo, const uint8_t* board5) {
    const auto& ct = ComboTable::get();
    const uint8_t a = ct.cards[combo][0], b = ct.cards[combo][1];
    for (int i = 0; i < 5; ++i)
        if (board5[i] == a || board5[i] == b) return -1;
    int seven[7];
    for (int i = 0; i < 5; ++i) seven[i] = Game::deck_to_two_plus_two(board5[i]);
    seven[5] = Game::deck_to_two_plus_two(a);
    seven[6] = Game::deck_to_two_plus_two(b);
    return Utility::LookupHandValue(seven);
}

void showdown_cfv(const std::vector<double>& opp, const uint8_t* board5,
                  double half_pot, std::vector<double>& cfv) {
    const auto& ct = ComboTable::get();
    std::vector<uint8_t> valid;
    board_valid(board5, 5, valid);

    // rank every valid combo once; sort combo indices by rank ascending
    std::vector<int> rank(kCombos, -1), order;
    order.reserve(kCombos);
    for (int i = 0; i < kCombos; ++i)
        if (valid[i]) {
            rank[i] = combo_rank(i, board5);
            order.push_back(i);
        }
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return rank[x] < rank[y]; });

    // totals for the lose-mass complement
    double S = 0.0, Sc[kCards] = {};
    for (int j : order) {
        S += opp[j];
        Sc[ct.cards[j][0]] += opp[j];
        Sc[ct.cards[j][1]] += opp[j];
    }

    cfv.assign(kCombos, 0.0);
    double W = 0.0, Wc[kCards] = {};  // running strictly-lower masses
    size_t g = 0;
    while (g < order.size()) {
        size_t e = g;
        while (e < order.size() && rank[order[e]] == rank[order[g]]) ++e;
        // tie-group sums
        double T = 0.0, Tc[kCards] = {};
        for (size_t k = g; k < e; ++k) {
            const int j = order[k];
            T += opp[j];
            Tc[ct.cards[j][0]] += opp[j];
            Tc[ct.cards[j][1]] += opp[j];
        }
        for (size_t k = g; k < e; ++k) {
            const int i = order[k];
            const int a = ct.cards[i][0], b = ct.cards[i][1];
            const double win = W - Wc[a] - Wc[b];          // i never in lower set
            const double tie = T - Tc[a] - Tc[b] + opp[i]; // i ties itself
            const double tot = S - Sc[a] - Sc[b] + opp[i];
            const double lose = tot - win - tie;
            cfv[i] = half_pot * (win - lose);
        }
        for (size_t k = g; k < e; ++k) {   // fold tie group into lower sums
            const int j = order[k];
            W += opp[j];
            Wc[ct.cards[j][0]] += opp[j];
            Wc[ct.cards[j][1]] += opp[j];
        }
        g = e;
    }
}

void compat_mass_brute(const std::vector<double>& opp,
                       const std::vector<uint8_t>& valid,
                       std::vector<double>& mass) {
    const auto& ct = ComboTable::get();
    mass.assign(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i) {
        if (!valid[i]) continue;
        const int a = ct.cards[i][0], b = ct.cards[i][1];
        double m = 0.0;
        for (int j = 0; j < kCombos; ++j) {
            if (!valid[j]) continue;
            const int c = ct.cards[j][0], d = ct.cards[j][1];
            if (c == a || c == b || d == a || d == b) continue;
            m += opp[j];
        }
        mass[i] = m;
    }
}

void showdown_cfv_brute(const std::vector<double>& opp, const uint8_t* board5,
                        double half_pot, std::vector<double>& cfv) {
    const auto& ct = ComboTable::get();
    std::vector<uint8_t> valid;
    board_valid(board5, 5, valid);
    std::vector<int> rank(kCombos, -1);
    for (int i = 0; i < kCombos; ++i)
        if (valid[i]) rank[i] = combo_rank(i, board5);
    cfv.assign(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i) {
        if (!valid[i]) continue;
        const int a = ct.cards[i][0], b = ct.cards[i][1];
        double v = 0.0;
        for (int j = 0; j < kCombos; ++j) {
            if (!valid[j] || j == i) continue;
            const int c = ct.cards[j][0], d = ct.cards[j][1];
            if (c == a || c == b || d == a || d == b) continue;
            if (rank[i] > rank[j]) v += opp[j];
            else if (rank[i] < rank[j]) v -= opp[j];
        }
        cfv[i] = half_pot * v;
    }
}

}  // namespace rebel_hunl
