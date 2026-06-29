// Exact-game tooling — C++ port of research/escher/solvers.py:
//   exact_br_value : recursive backward-induction best response (the exact
//                    exploitability metric)
//   exploitability : BR0 + BR1 vs an average strategy (0 at Nash)
//   CFRPlus        : tabular RM⁺ + linear averaging — the validation reference
//                    (INCLUDES the chance/deal-probability weighting fix)
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "games.h"

namespace escher {

// A tabular strategy: infoset key -> per-action probability (indexed by action
// id, size n_actions; illegal actions hold 0).
using Strategy = std::unordered_map<std::string, std::vector<double>>;

inline double walk_value(const Game& g, const std::string& h, const Cards& c,
                         const Strategy& strat) {
    if (g.is_terminal(h)) return g.terminal_util_p0(h, c);
    const auto& probs = strat.at(g.infoset_key(h, c));
    double v = 0.0;
    for (int a : g.legal_actions(h)) {
        double pa = probs[a];
        if (pa == 0.0) continue;
        v += pa * walk_value(g, g.step(h, a), c, strat);
    }
    return v;
}

inline double expected_value(const Game& g, const Strategy& strat) {
    double v = 0.0;
    for (const auto& d : g.deals())
        v += d.prob * walk_value(g, "", d.cards, strat);
    return v;
}

// Recursive exact best response (backward induction over infosets, deepest
// first). Validated == brute force on Kuhn in the Python prototype.
inline double exact_br_value(const Game& g, int br_player,
                             const Strategy& sigma) {
    struct Member { std::string h; Cards c; double opp_reach; };
    std::unordered_map<std::string, std::vector<Member>> members;

    std::function<void(const std::string&, const Cards&, double)> fwd =
        [&](const std::string& h, const Cards& c, double opp) {
            if (g.is_terminal(h)) return;
            int p = g.current_player(h);
            std::string I = g.infoset_key(h, c);
            auto legal = g.legal_actions(h);
            if (p == br_player) {
                members[I].push_back({h, c, opp});
                for (int a : legal) fwd(g.step(h, a), c, opp);
            } else {
                const auto& sig = sigma.at(I);
                for (int a : legal) fwd(g.step(h, a), c, opp * sig[a]);
            }
        };
    for (const auto& d : g.deals()) fwd("", d.cards, d.prob);

    std::unordered_map<std::string, int> br_action;

    std::function<double(const std::string&, const Cards&)> value =
        [&](const std::string& h, const Cards& c) -> double {
            if (g.is_terminal(h)) {
                double u = g.terminal_util_p0(h, c);
                return br_player == 0 ? u : -u;
            }
            int p = g.current_player(h);
            std::string I = g.infoset_key(h, c);
            if (p == br_player) return value(g.step(h, br_action.at(I)), c);
            const auto& sig = sigma.at(I);
            double v = 0.0;
            for (int a : g.legal_actions(h)) v += sig[a] * value(g.step(h, a), c);
            return v;
        };

    // solve deepest infosets first (a br node's children are strictly deeper)
    std::vector<std::string> keys;
    keys.reserve(members.size());
    for (auto& kv : members) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end(), [&](const std::string& a, const std::string& b) {
        return members[a][0].h.size() > members[b][0].h.size();
    });
    for (const auto& I : keys) {
        auto legal = g.legal_actions(members[I][0].h);
        double best = -1e18; int besta = legal[0];
        for (int a : legal) {
            double q = 0.0;
            for (const auto& m : members[I]) q += m.opp_reach * value(g.step(m.h, a), m.c);
            if (q > best) { best = q; besta = a; }
        }
        br_action[I] = besta;
    }

    double total = 0.0;
    for (const auto& d : g.deals()) total += d.prob * value("", d.cards);
    return total;
}

inline double exploitability(const Game& g, const Strategy& avg) {
    return exact_br_value(g, 0, avg) + exact_br_value(g, 1, avg);
}

// ─── Tabular CFR+ (RM⁺ + linear averaging) — validation reference ───────────
class CFRPlus {
public:
    explicit CFRPlus(const Game& g) : g_(g), A_(g.n_actions()) {}

    void iterate(int n_iters) {
        for (int it = 0; it < n_iters; ++it) {
            ++t_;
            for (const auto& d : g_.deals()) cfr("", d.cards, 1.0, 1.0, d.prob);
        }
    }

    Strategy average_strategy() const {
        Strategy avg;
        for (const auto& kv : strat_sum_) {
            const auto& legal = infoset_actions_.at(kv.first);
            double tot = 0.0;
            for (int a : legal) tot += kv.second[a];
            std::vector<double> p(A_, 0.0);
            if (tot > 0) for (int a : legal) p[a] = kv.second[a] / tot;
            else for (int a : legal) p[a] = 1.0 / legal.size();
            avg[kv.first] = p;
        }
        return avg;
    }

private:
    std::vector<double> strategy(const std::string& I, const std::vector<int>& legal) {
        auto it = regret_.find(I);
        std::vector<double> s(A_, 0.0);
        double sum = 0.0;
        if (it != regret_.end())
            for (int a : legal) { double r = it->second[a]; if (r > 0) { s[a] = r; sum += r; } }
        if (sum > 0) for (int a : legal) s[a] /= sum;
        else for (int a : legal) s[a] = 1.0 / legal.size();
        return s;
    }

    double cfr(const std::string& h, const Cards& c, double p0, double p1, double q) {
        if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
        int player = g_.current_player(h);
        auto legal = g_.legal_actions(h);
        std::string I = g_.infoset_key(h, c);
        infoset_actions_[I] = legal;
        if (!regret_.count(I)) regret_[I] = std::vector<double>(A_, 0.0);
        if (!strat_sum_.count(I)) strat_sum_[I] = std::vector<double>(A_, 0.0);
        auto sigma = strategy(I, legal);

        std::vector<double> util_a(A_, 0.0);
        double node = 0.0;
        for (int a : legal) {
            util_a[a] = cfr(g_.step(h, a), c, player == 0 ? p0 * sigma[a] : p0,
                            player == 1 ? p1 * sigma[a] : p1, q);
            node += sigma[a] * util_a[a];
        }
        // q = chance/deal reach — folded into BOTH the cf reach and the avg weight
        double cf = q * (player == 0 ? p1 : p0);
        double own = q * (player == 0 ? p0 : p1);
        double sign = player == 0 ? 1.0 : -1.0;
        for (int a : legal) {
            double r = sign * (util_a[a] - node) * cf;
            regret_[I][a] = std::max(regret_[I][a] + r, 0.0);
            strat_sum_[I][a] += t_ * own * sigma[a];
        }
        return node;
    }

    const Game& g_;
    int A_;
    long t_ = 0;
    std::unordered_map<std::string, std::vector<double>> regret_, strat_sum_;
    std::unordered_map<std::string, std::vector<int>> infoset_actions_;
};

}  // namespace escher
