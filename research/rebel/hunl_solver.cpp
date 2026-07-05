#include "hunl_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rebel_hunl {

namespace {
constexpr double kTiny = 1e-12;

void normalize_range(std::vector<double>& r, const std::vector<uint8_t>& valid) {
    r.resize(kCombos, 0.0);
    double s = 0.0;
    for (int i = 0; i < kCombos; ++i) {
        if (!valid[i]) r[i] = 0.0;
        s += r[i];
    }
    if (s > kTiny) {
        for (double& x : r) x /= s;
        return;
    }
    int m = 0;
    for (int i = 0; i < kCombos; ++i) m += valid[i];
    for (int i = 0; i < kCombos; ++i) r[i] = valid[i] ? 1.0 / m : 0.0;
}

void mask_card(std::vector<double>& r, uint8_t c) {
    const auto& ct = ComboTable::get();
    for (int i = 0; i < kCombos; ++i)
        if (ct.cards[i][0] == c || ct.cards[i][1] == c) r[i] = 0.0;
}
}  // namespace

HunlSolver::HunlSolver(poker_ppo::PokerEnvironment& env, HunlPBS root,
                       HunlValueOracle* oracle,
                       std::vector<int> allowed_actions,
                       const uint8_t* board_override, int nb_override)
    : env_(env), root_(std::move(root)), oracle_(oracle),
      allowed_(std::move(allowed_actions)) {
    root_round_ = env_.round();
    if (board_override != nullptr) {
        nb_root_ = nb_override;
        for (int i = 0; i < nb_root_; ++i) board_[i] = board_override[i];
    } else {
        nb_root_ = env_.community_count();
        for (int i = 0; i < nb_root_; ++i)
            board_[i] = static_cast<uint8_t>(env_.community_card(i));
    }
    board_valid(board_.data(), nb_root_, valid_);
    normalize_range(root_.r0, valid_);
    normalize_range(root_.r1, valid_);
    build();
    leaf_v_.resize(nodes_.size());
}

int HunlSolver::build() {
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    {
        Node& nd = nodes_.back();
        nd.contrib = {
            static_cast<int>(env_.game_config().initial_stack) - env_.stack(0),
            static_cast<int>(env_.game_config().initial_stack) - env_.stack(1)};
        if (env_.is_terminal()) {
            if (env_.terminal_was_fold())
                nd.kind = Node::Fold;
            else
                nd.kind = env_.community_count() >= 5 ? Node::Showdown
                                                      : Node::AllinShowdown;
            return id;
        }
        if (env_.round() > root_round_) {
            nd.kind = Node::StreetEnd;
            leaf_ids_.push_back(id);
            return id;
        }
        nd.kind = Node::Decision;
        nd.player = env_.current_player();
    }
    auto mask = env_.legal_action_mask();
    auto ma = mask.accessor<float, 1>();
    std::vector<int> acts;
    for (int a = 0; a < static_cast<int>(ma.size(0)); ++a) {
        if (ma[a] < 0.5f) continue;
        if (!allowed_.empty() &&
            std::find(allowed_.begin(), allowed_.end(), a) == allowed_.end())
            continue;
        acts.push_back(a);
    }
    if (acts.empty()) acts.push_back(1);   // call is always legal
    std::vector<int> children;
    children.reserve(acts.size());
    for (int a : acts) {
        env_.push_state();
        env_.step(a);
        const int cid = build();   // may reallocate nodes_
        if (nodes_[cid].kind == Node::Fold)
            nodes_[cid].player = nodes_[id].player;   // the folder
        nodes_[cid].path = nodes_[id].path;
        nodes_[cid].path.push_back(a);
        env_.pop_state();
        children.push_back(cid);
    }
    Node& nd = nodes_[id];
    nd.acts = std::move(acts);
    nd.child = std::move(children);
    const int A = static_cast<int>(nd.acts.size());
    nd.regret.assign(static_cast<size_t>(kCombos) * A, 0.0);
    nd.cum_strat.assign(static_cast<size_t>(kCombos) * A, 0.0);
    return id;
}

std::vector<double> HunlSolver::policy_row(const Node& nd, int combo,
                                           bool average) const {
    const int A = static_cast<int>(nd.acts.size());
    std::vector<double> p(A, 0.0);
    const std::vector<double>& src = average ? nd.cum_strat : nd.regret;
    double s = 0.0;
    for (int k = 0; k < A; ++k) {
        double v = src[static_cast<size_t>(combo) * A + k];
        if (v < 0.0) v = 0.0;   // regrets are RM⁺-clamped anyway; belt+braces
        p[k] = v;
        s += v;
    }
    if (s > kTiny)
        for (double& x : p) x /= s;
    else
        for (double& x : p) x = 1.0 / A;
    return p;
}

std::vector<double> HunlSolver::avg_policy(int node, int combo) const {
    return policy_row(nodes_[node], combo, /*average=*/true);
}

void HunlSolver::policies_into(const Node& nd, bool average,
                               std::vector<double>& out) const {
    const int A = static_cast<int>(nd.acts.size());
    const std::vector<double>& src = average ? nd.cum_strat : nd.regret;
    out.resize(static_cast<size_t>(kCombos) * A);
    for (int x = 0; x < kCombos; ++x) {
        double s = 0.0;
        const size_t off = static_cast<size_t>(x) * A;
        for (int k = 0; k < A; ++k) {
            double v = src[off + k];
            if (v < 0.0) v = 0.0;
            out[off + k] = v;
            s += v;
        }
        if (s > kTiny)
            for (int k = 0; k < A; ++k) out[off + k] /= s;
        else
            for (int k = 0; k < A; ++k) out[off + k] = 1.0 / A;
    }
}

void HunlSolver::reaches_to(int target, bool average, std::vector<double>& r0,
                            std::vector<double>& r1) const {
    r0 = root_.r0;
    r1 = root_.r1;
    int i = 0;
    for (int a : nodes_[target].path) {
        const Node& nd = nodes_[i];
        int k = 0;
        while (nd.acts[k] != a) ++k;
        std::vector<double>& mine = nd.player == 0 ? r0 : r1;
        for (int c = 0; c < kCombos; ++c)
            if (mine[c] > 0.0) mine[c] *= policy_row(nd, c, average)[k];
        i = nd.child[k];
    }
}

void HunlSolver::refresh_leaves() {
    if (oracle_ == nullptr || leaf_ids_.empty()) return;
    for (int L : leaf_ids_) {
        auto& per = leaf_v_[L];
        per.assign(kCards, {});
        std::vector<double> r0, r1;
        reaches_to(L, /*average=*/true, r0, r1);   // CFR-AVG leaf beliefs
        env_.push_state();
        for (int a : nodes_[L].path) env_.step(a);
        for (int c = 0; c < kCards; ++c) {
            bool on_board = false;
            for (int b = 0; b < nb_root_; ++b)
                if (board_[b] == c) on_board = true;
            if (on_board) continue;
            HunlPBS beta;
            beta.r0 = r0;
            beta.r1 = r1;
            mask_card(beta.r0, static_cast<uint8_t>(c));
            mask_card(beta.r1, static_cast<uint8_t>(c));
            std::array<uint8_t, 5> nb = board_;
            nb[nb_root_] = static_cast<uint8_t>(c);
            oracle_->value(env_, nb.data(), nb_root_ + 1, beta, per[c]);
        }
        env_.pop_state();
    }
}

std::vector<double> HunlSolver::walk(int i, int upd, int t, bool update,
                                     std::vector<double>& my_reach,
                                     std::vector<double>& opp_reach) {
    Node& nd = nodes_[i];
    std::vector<double> cfv(kCombos, 0.0);
    const auto& ct = ComboTable::get();

    switch (nd.kind) {
    case Node::Fold: {
        const double u = (upd == nd.player)
            ? -static_cast<double>(nd.contrib[upd])
            : static_cast<double>(nd.contrib[1 - upd]);
        fold_cfv(opp_reach, valid_, u, cfv);
        return cfv;
    }
    case Node::Showdown: {
        showdown_cfv(opp_reach, board_.data(),
                     static_cast<double>(nd.contrib[0]), cfv);
        return cfv;
    }
    case Node::AllinShowdown: {
        if (nb_root_ + 1 < 5) {
            std::fprintf(stderr, "AllinShowdown with %d cards to come is a "
                         "later stage (flop/preflop subgames)\n",
                         5 - nb_root_);
            std::abort();
        }
        const double n_rem = 52.0 - nb_root_ - 4.0;
        std::vector<double> part;
        for (int c = 0; c < kCards; ++c) {
            bool used = false;
            for (int b = 0; b < nb_root_; ++b)
                if (board_[b] == c) used = true;
            if (used) continue;
            std::array<uint8_t, 5> nb = board_;
            nb[nb_root_] = static_cast<uint8_t>(c);
            std::vector<double> opp_c = opp_reach;
            mask_card(opp_c, static_cast<uint8_t>(c));
            showdown_cfv(opp_c, nb.data(),
                         static_cast<double>(nd.contrib[0]), part);
            for (int x = 0; x < kCombos; ++x) {
                if (!valid_[x]) continue;
                if (ct.cards[x][0] == c || ct.cards[x][1] == c) continue;
                cfv[x] += part[x] / n_rem;
            }
        }
        return cfv;
    }
    case Node::StreetEnd: {
        if (leaf_v_[i].empty()) return cfv;   // no oracle: values 0 (unused)
        const double n_rem = 52.0 - nb_root_ - 4.0;
        for (int c = 0; c < kCards; ++c) {
            const auto& slot = leaf_v_[i][c];
            const auto& v = slot[upd];
            if (v.empty()) continue;
            double S = 0.0, Sc[kCards] = {};
            for (int j = 0; j < kCombos; ++j) {
                if (opp_reach[j] <= 0.0) continue;
                const int a = ct.cards[j][0], b = ct.cards[j][1];
                if (a == c || b == c) continue;
                S += opp_reach[j];
                Sc[a] += opp_reach[j];
                Sc[b] += opp_reach[j];
            }
            for (int x = 0; x < kCombos; ++x) {
                if (!valid_[x]) continue;
                const int a = ct.cards[x][0], b = ct.cards[x][1];
                if (a == c || b == c) continue;
                const double mass = S - Sc[a] - Sc[b] + opp_reach[x];
                if (mass > 0.0) cfv[x] += v[x] * mass / n_rem;
            }
        }
        return cfv;
    }
    case Node::Decision:
        break;
    }

    const int A = static_cast<int>(nd.acts.size());
    const bool use_avg = !update;
    std::vector<double> sig;
    policies_into(nd, use_avg, sig);   // flat [combo*A+k]
    if (nd.player == upd) {
        std::vector<std::vector<double>> cfv_a(A);
        for (int k = 0; k < A; ++k) {
            std::vector<double> child_reach(kCombos, 0.0);
            for (int x = 0; x < kCombos; ++x)
                if (valid_[x] && my_reach[x] > 0.0)
                    child_reach[x] =
                        my_reach[x] * sig[static_cast<size_t>(x) * A + k];
            cfv_a[k] =
                walk(nd.child[k], upd, t, update, child_reach, opp_reach);
        }
        for (int x = 0; x < kCombos; ++x) {
            if (!valid_[x]) continue;
            const size_t off = static_cast<size_t>(x) * A;
            double v = 0.0;
            for (int k = 0; k < A; ++k) v += sig[off + k] * cfv_a[k][x];
            cfv[x] = v;
            if (!update) continue;
            for (int k = 0; k < A; ++k) {
                double& r = nd.regret[off + k];
                r += cfv_a[k][x] - v;
                if (r < 0.0) r = 0.0;   // RM⁺
                nd.cum_strat[off + k] +=
                    static_cast<double>(t) * my_reach[x] * sig[off + k];
            }
        }
        return cfv;
    }

    for (int k = 0; k < A; ++k) {
        std::vector<double> child_opp(kCombos, 0.0);
        for (int x = 0; x < kCombos; ++x)
            if (valid_[x] && opp_reach[x] > 0.0)
                child_opp[x] =
                    opp_reach[x] * sig[static_cast<size_t>(x) * A + k];
        auto cv = walk(nd.child[k], upd, t, update, my_reach, child_opp);
        for (int x = 0; x < kCombos; ++x) cfv[x] += cv[x];
    }
    return cfv;
}

void HunlSolver::iterate(int t) {
    if (t == 1 || refresh_every <= 1 || t % refresh_every == 0)
        refresh_leaves();
    for (int upd = 0; upd < 2; ++upd) {
        std::vector<double> mine = upd == 0 ? root_.r0 : root_.r1;
        std::vector<double> opp = upd == 0 ? root_.r1 : root_.r0;
        walk(0, upd, t, /*update=*/true, mine, opp);
    }
}

std::vector<double> HunlSolver::br_walk(int i, int p,
                                        std::vector<double>& opp_reach) {
    Node& nd = nodes_[i];
    if (nd.kind != Node::Decision) {
        std::vector<double> dummy_my(kCombos, 0.0);
        return walk(i, p, /*t=*/0, /*update=*/false, dummy_my, opp_reach);
    }
    const int A = static_cast<int>(nd.acts.size());
    std::vector<double> cfv(kCombos, 0.0);
    if (nd.player == p) {
        bool first = true;
        for (int k = 0; k < A; ++k) {
            auto cv = br_walk(nd.child[k], p, opp_reach);
            if (first) {
                cfv = cv;
                first = false;
            } else {
                for (int x = 0; x < kCombos; ++x)
                    if (cv[x] > cfv[x]) cfv[x] = cv[x];
            }
        }
        return cfv;
    }
    for (int k = 0; k < A; ++k) {
        std::vector<double> child_opp(kCombos, 0.0);
        for (int x = 0; x < kCombos; ++x)
            if (valid_[x] && opp_reach[x] > 0.0)
                child_opp[x] =
                    opp_reach[x] * policy_row(nd, x, /*average=*/true)[k];
        auto cv = br_walk(nd.child[k], p, child_opp);
        for (int x = 0; x < kCombos; ++x) cfv[x] += cv[x];
    }
    return cfv;
}

double HunlSolver::exploitability() {
    refresh_leaves();   // BR terminals read final-average leaf values
    double total = 0.0, joint = 0.0;
    std::vector<double> mass;
    compat_mass(root_.r1, valid_, mass);
    for (int i = 0; i < kCombos; ++i) joint += root_.r0[i] * mass[i];
    for (int p = 0; p < 2; ++p) {
        std::vector<double> opp = p == 0 ? root_.r1 : root_.r0;
        auto br = br_walk(0, p, opp);
        const auto& own = p == 0 ? root_.r0 : root_.r1;
        for (int x = 0; x < kCombos; ++x)
            if (valid_[x]) total += own[x] * br[x];
    }
    return joint > kTiny ? total / joint : 0.0;   // chips/hand; → 0 at Nash
}

void HunlSolver::root_values(std::array<std::vector<double>, 2>& v,
                             std::array<std::vector<double>, 2>& mask) {
    refresh_leaves();
    for (int p = 0; p < 2; ++p) {
        std::vector<double> mine = p == 0 ? root_.r0 : root_.r1;
        std::vector<double> opp = p == 0 ? root_.r1 : root_.r0;
        auto cfv = walk(0, p, /*t=*/0, /*update=*/false, mine, opp);
        std::vector<double> m;
        compat_mass(opp, valid_, m);
        v[p].assign(kCombos, 0.0);
        mask[p].assign(kCombos, 0.0);
        const auto& own = p == 0 ? root_.r0 : root_.r1;
        for (int x = 0; x < kCombos; ++x) {
            if (!valid_[x] || own[x] <= kTiny || m[x] <= kTiny) continue;
            v[p][x] = cfv[x] / m[x];
            mask[p][x] = 1.0;
        }
    }
}

void ExactStreetOracle::value(poker_ppo::PokerEnvironment& env,
                              const uint8_t* board, int nb,
                              const HunlPBS& beta,
                              std::array<std::vector<double>, 2>& out) {
    HunlSolver s(env, beta, /*oracle=*/nullptr, allowed_, board, nb);
    for (int t = 1; t <= iters_; ++t) s.iterate(t);
    std::array<std::vector<double>, 2> mask;
    s.root_values(out, mask);
}

}  // namespace rebel_hunl
