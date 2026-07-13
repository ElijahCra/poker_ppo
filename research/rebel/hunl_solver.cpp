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
    build({});
    leaf_v_.resize(nodes_.size());
}

int HunlSolver::build(std::vector<int> path) {
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    {
        Node& nd = nodes_.back();
        nd.path = std::move(path);   // at CREATION — children copy from it
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
    // Multi-street subgames (flop and earlier): prune RAISE lines whose
    // call would end the hand as an all-in with >1 card to come — beyond
    // the exact runout enumeration (walk's abort remains the backstop for
    // call-created ones after an off-tree real shove). Pruning must be at
    // the raise, never the call: dropping the call would teach the model
    // that shoves force folds. Inert for turn/river roots.
    if (nb_root_ + 1 < 5) {
        std::vector<int> kept;
        kept.reserve(acts.size());
        for (int a : acts) {
            bool bad = false;
            if (a >= 2) {
                env_.push_state();
                env_.step(a);
                if (!env_.is_terminal()) {
                    env_.push_state();
                    env_.step(1);
                    bad = env_.is_terminal() &&
                          !env_.terminal_was_fold() &&
                          env_.community_count() < 5;
                    env_.pop_state();
                }
                env_.pop_state();
            }
            if (!bad) kept.push_back(a);
        }
        if (!kept.empty()) acts = std::move(kept);
    }
    std::vector<int> children;
    children.reserve(acts.size());
    for (int a : acts) {
        env_.push_state();
        env_.step(a);
        std::vector<int> child_path = nodes_[id].path;
        child_path.push_back(a);
        const int cid = build(std::move(child_path));  // may realloc nodes_
        if (nodes_[cid].kind == Node::Fold)
            nodes_[cid].player = nodes_[id].player;   // the folder
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

const RiverEval& HunlSolver::eval_for_root() {
    if (!river_eval_)
        river_eval_ = std::make_unique<RiverEval>(board_.data());
    return *river_eval_;
}

const RiverEval& HunlSolver::eval_for_runout(uint8_t c) {
    if (!runout_eval_[c]) {
        std::array<uint8_t, 5> nb = board_;
        nb[nb_root_] = c;
        runout_eval_[c] = std::make_unique<RiverEval>(nb.data());
    }
    return *runout_eval_[c];
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
    if (nb_root_ == 0) {
        // preflop: fixed sampled-flop leaf set, drawn once
        if (pf_flops_.empty()) {
            std::mt19937_64 rng(pf_seed);
            std::vector<uint8_t> deck(kCards);
            for (int c = 0; c < kCards; ++c)
                deck[static_cast<size_t>(c)] = static_cast<uint8_t>(c);
            for (int k = 0; k < pf_samples; ++k) {
                for (int j = 0; j < 3; ++j)
                    std::swap(deck[static_cast<size_t>(j)],
                              deck[static_cast<size_t>(
                                  j + rng() % (kCards - j))]);
                pf_flops_.push_back({deck[0], deck[1], deck[2]});
            }
            const auto& ct = ComboTable::get();
            pf_cnt_.assign(kCombos, 0.0);
            for (int x = 0; x < kCombos; ++x)
                for (const auto& f : pf_flops_) {
                    bool hit = false;
                    for (int j = 0; j < 3; ++j)
                        if (f[j] == ct.cards[x][0] ||
                            f[j] == ct.cards[x][1])
                            hit = true;
                    if (!hit) pf_cnt_[x] += 1.0;
                }
            leaf_v_pf_.resize(nodes_.size());
        }
        for (int L : leaf_ids_) {
            std::vector<double> r0, r1;
            reaches_to(L, /*average=*/true, r0, r1);
            env_.push_state();
            for (int a : nodes_[L].path) env_.step(a);
            std::vector<HunlPBS> betas;
            betas.reserve(pf_flops_.size());
            for (const auto& f : pf_flops_) {
                HunlPBS beta;
                beta.r0 = r0;
                beta.r1 = r1;
                for (int j = 0; j < 3; ++j) {
                    mask_card(beta.r0, f[j]);
                    mask_card(beta.r1, f[j]);
                }
                betas.push_back(std::move(beta));
            }
            oracle_->value_boards(env_, pf_flops_, betas, leaf_v_pf_[L]);
            env_.pop_state();
        }
        return;
    }
    for (int L : leaf_ids_) {
        auto& per = leaf_v_[L];
        per.assign(kCards, {});
        std::vector<double> r0, r1;
        reaches_to(L, /*average=*/true, r0, r1);   // CFR-AVG leaf beliefs
        env_.push_state();
        for (int a : nodes_[L].path) env_.step(a);
        std::vector<uint8_t> cards;
        std::vector<HunlPBS> betas;
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
            cards.push_back(static_cast<uint8_t>(c));
            betas.push_back(std::move(beta));
        }
        std::vector<std::array<std::vector<double>, 2>> outs;
        oracle_->value_batch(env_, board_.data(), nb_root_, cards, betas,
                             outs);
        for (size_t k = 0; k < cards.size(); ++k)
            per[cards[k]] = std::move(outs[k]);
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
        eval_for_root().cfv(opp_reach, static_cast<double>(nd.contrib[0]),
                            cfv);
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
            std::vector<double> opp_c = opp_reach;
            mask_card(opp_c, static_cast<uint8_t>(c));
            eval_for_runout(static_cast<uint8_t>(c))
                .cfv(opp_c, static_cast<double>(nd.contrib[0]), part);
            for (int x = 0; x < kCombos; ++x) {
                if (!valid_[x]) continue;
                if (ct.cards[x][0] == c || ct.cards[x][1] == c) continue;
                cfv[x] += part[x] / n_rem;
            }
        }
        return cfv;
    }
    case Node::StreetEnd: {
        if (nb_root_ == 0) {
            // preflop: aggregate the sampled flops; a combo averages only
            // over flops its own cards don't knock out (pf_cnt_)
            if (leaf_v_pf_.empty() || leaf_v_pf_[static_cast<size_t>(i)]
                                          .empty())
                return cfv;
            const auto& per = leaf_v_pf_[static_cast<size_t>(i)];
            for (size_t f = 0; f < pf_flops_.size(); ++f) {
                const auto& fl = pf_flops_[f];
                const auto& v = per[f][static_cast<size_t>(upd)];
                if (v.empty()) continue;
                double S = 0.0, Sc[kCards] = {};
                for (int j = 0; j < kCombos; ++j) {
                    if (opp_reach[j] <= 0.0) continue;
                    const int a = ct.cards[j][0], b = ct.cards[j][1];
                    if (a == fl[0] || a == fl[1] || a == fl[2] ||
                        b == fl[0] || b == fl[1] || b == fl[2])
                        continue;
                    S += opp_reach[j];
                    Sc[a] += opp_reach[j];
                    Sc[b] += opp_reach[j];
                }
                for (int x = 0; x < kCombos; ++x) {
                    if (!valid_[x] || pf_cnt_[x] <= 0.0) continue;
                    const int a = ct.cards[x][0], b = ct.cards[x][1];
                    if (a == fl[0] || a == fl[1] || a == fl[2] ||
                        b == fl[0] || b == fl[1] || b == fl[2])
                        continue;
                    const double mass = S - Sc[a] - Sc[b] + opp_reach[x];
                    if (mass > 0.0) cfv[x] += v[x] * mass / pf_cnt_[x];
                }
            }
            return cfv;
        }
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

void HunlSolver::enable_gadget(int opp, const std::vector<double>& alt,
                               double mix) {
    gadget_opp_ = opp;
    // entry prior: tracked (already normalized) range floored with uniform
    std::vector<double>& r = opp == 0 ? root_.r0 : root_.r1;
    int nvalid = 0;
    for (int i = 0; i < kCombos; ++i)
        if (valid_[i]) ++nvalid;
    for (int i = 0; i < kCombos; ++i)
        r[i] = valid_[i]
            ? (1.0 - mix) * r[i] + mix / std::max(1, nvalid)
            : 0.0;
    // Regrets compare mass-weighted (counterfactual) payoffs: entering
    // yields the walk's unnormalized CFV (weighted by OUR compatible
    // reach); Terminate must be weighted identically. Our root range is
    // fixed across iterations, so the weights are too.
    const std::vector<double>& us = opp == 0 ? root_.r1 : root_.r0;
    std::vector<double> m;
    compat_mass(us, valid_, m);
    gd_alt_w_.assign(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i)
        if (valid_[i]) gd_alt_w_[i] = alt[i] * m[i];
    gd_rT_.assign(kCombos, 0.0);
    gd_rF_.assign(kCombos, 0.0);
    gd_pF_.assign(kCombos, 1.0);   // optimistic: enter until told otherwise
    gd_cumF_.assign(kCombos, 0.0);
    gd_cumW_ = 0.0;
}

void HunlSolver::iterate(int t) {
    if (t == 1 || refresh_every <= 1 || t % refresh_every == 0)
        refresh_leaves();
    if (gadget_opp_ < 0) {
        for (int upd = 0; upd < 2; ++upd) {
            std::vector<double> mine = upd == 0 ? root_.r0 : root_.r1;
            std::vector<double> opp = upd == 0 ? root_.r1 : root_.r0;
            auto cfv = walk(0, upd, t, /*update=*/true, mine, opp);
            if (track_root_) {
                if (rv_sum_[upd].empty()) rv_sum_[upd].assign(kCombos, 0.0);
                for (int x = 0; x < kCombos; ++x) rv_sum_[upd][x] += cfv[x];
                if (upd == 1) ++rv_n_;
            }
        }
        return;
    }
    // gadgeted: the opponent's reach into the subgame is prior · pF (this
    // iteration's Follow probability), for BOTH update passes; after the
    // opponent's own pass, regret-match T vs F per combo on its CFVs.
    for (int upd = 0; upd < 2; ++upd) {
        std::vector<double> mine = upd == 0 ? root_.r0 : root_.r1;
        std::vector<double> opp = upd == 0 ? root_.r1 : root_.r0;
        auto& gadget_side = upd == gadget_opp_ ? mine : opp;
        for (int i = 0; i < kCombos; ++i) gadget_side[i] *= gd_pF_[i];
        auto cfv = walk(0, upd, t, /*update=*/true, mine, opp);
        if (upd != gadget_opp_) continue;
        for (int j = 0; j < kCombos; ++j) {
            if (!valid_[j]) continue;
            const double vF = cfv[j];                 // enter (mass-weighted)
            const double vT = gd_alt_w_[j];           // terminate
            const double v  = gd_pF_[j] * vF + (1.0 - gd_pF_[j]) * vT;
            gd_rF_[j] = std::max(0.0, gd_rF_[j] + vF - v);   // RM⁺
            gd_rT_[j] = std::max(0.0, gd_rT_[j] + vT - v);
            gd_cumF_[j] += static_cast<double>(t) * gd_pF_[j];
        }
        gd_cumW_ += static_cast<double>(t);
        for (int j = 0; j < kCombos; ++j) {
            if (!valid_[j]) continue;
            const double s = gd_rF_[j] + gd_rT_[j];
            gd_pF_[j] = s > 0.0 ? gd_rF_[j] / s : 1.0;
        }
    }
}

bool HunlSolver::avg_root_values(std::array<std::vector<double>, 2>& v,
                                 std::array<std::vector<double>, 2>& mask)
    const {
    if (!track_root_ || rv_n_ <= 0) return false;
    for (int p = 0; p < 2; ++p) {
        if (rv_sum_[p].empty()) return false;
        const auto& opp = p == 0 ? root_.r1 : root_.r0;
        const auto& own = p == 0 ? root_.r0 : root_.r1;
        std::vector<double> m;
        compat_mass(opp, valid_, m);
        v[p].assign(kCombos, 0.0);
        mask[p].assign(kCombos, 0.0);
        for (int x = 0; x < kCombos; ++x) {
            if (!valid_[x] || own[x] <= kTiny || m[x] <= 1e-6) continue;
            v[p][x] = rv_sum_[p][x] / static_cast<double>(rv_n_) / m[x];
            mask[p][x] = 1.0;
        }
    }
    return true;
}

void HunlSolver::beliefs_at(int node, std::vector<double>& r0,
                            std::vector<double>& r1) const {
    reaches_to(node, /*average=*/true, r0, r1);
    if (gadget_opp_ >= 0 && gd_cumW_ > 0.0) {
        std::vector<double>& g = gadget_opp_ == 0 ? r0 : r1;
        for (int i = 0; i < kCombos; ++i)
            g[i] *= gd_cumF_[i] / gd_cumW_;
    }
}

void HunlSolver::values_at(int node, std::array<std::vector<double>, 2>& v,
                           std::array<std::vector<double>, 2>& mask) {
    std::vector<double> r0, r1;
    beliefs_at(node, r0, r1);
    for (int p = 0; p < 2; ++p) {
        std::vector<double> mine = p == 0 ? r0 : r1;
        std::vector<double> opp = p == 0 ? r1 : r0;
        auto cfv = walk(node, p, /*t=*/0, /*update=*/false, mine, opp);
        std::vector<double> m;
        compat_mass(opp, valid_, m);
        // normalize opposing mass so per-combo values are conditional
        // expectations (same convention as root_values / the net)
        double s = 0.0;
        for (int i = 0; i < kCombos; ++i)
            if (valid_[i]) s += opp[i];
        v[p].assign(kCombos, 0.0);
        mask[p].assign(kCombos, 0.0);
        if (s <= kTiny) continue;
        for (int x = 0; x < kCombos; ++x) {
            if (!valid_[x] || m[x] / s <= 1e-6) continue;
            v[p][x] = cfv[x] / m[x];
            mask[p][x] = 1.0;
        }
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

std::vector<double> HunlSolver::best_response(int p,
                                              std::vector<double> opp_reach) {
    return br_walk(0, p, opp_reach);
}

double HunlSolver::exploitability_composed(int t_river) {
    TORCH_CHECK(nb_root_ > 0,
                "composed BR assumes single-card chance (not preflop)");
    const auto& ct = ComboTable::get();
    const double n_rem = 52.0 - nb_root_ - 4.0;
    // composed BR leaf values per (StreetEnd, player), aggregated over cards
    std::vector<std::array<std::vector<double>, 2>> br_leaf(nodes_.size());
    for (int L : leaf_ids_) {
        br_leaf[L][0].assign(kCombos, 0.0);
        br_leaf[L][1].assign(kCombos, 0.0);
        std::vector<double> r0, r1;
        reaches_to(L, /*average=*/true, r0, r1);
        env_.push_state();
        for (int a : nodes_[L].path) env_.step(a);
        for (int c = 0; c < kCards; ++c) {
            bool on_board = false;
            for (int b = 0; b < nb_root_; ++b)
                if (board_[b] == c) on_board = true;
            if (on_board) continue;
            std::array<uint8_t, 5> nb = board_;
            nb[nb_root_] = static_cast<uint8_t>(c);
            // the agent's deployment rule: re-solve at normalized avg beliefs
            HunlPBS beta;
            beta.r0 = r0;
            beta.r1 = r1;
            mask_card(beta.r0, static_cast<uint8_t>(c));
            mask_card(beta.r1, static_cast<uint8_t>(c));
            HunlSolver rs(env_, beta, nullptr, allowed_, nb.data(),
                          nb_root_ + 1);
            for (int t = 1; t <= t_river; ++t) rs.iterate(t);
            // full-game BR: deviate freely on the river too, opponent pinned
            // to the composed average; opp reach = UNNORMALIZED turn reach
            for (int p = 0; p < 2; ++p) {
                std::vector<double> opp = p == 0 ? r1 : r0;
                mask_card(opp, static_cast<uint8_t>(c));
                auto br = rs.best_response(p, std::move(opp));
                auto& acc = br_leaf[L][p];
                for (int x = 0; x < kCombos; ++x) {
                    if (ct.cards[x][0] == c || ct.cards[x][1] == c) continue;
                    acc[x] += br[x] / n_rem;
                }
            }
        }
        env_.pop_state();
    }

    // two-street BR walk: StreetEnd returns the composed river BR values
    // (the walk's opponent reach at a leaf equals reaches_to(avg) — the
    // opponent is pinned to the average on both streets)
    std::function<std::vector<double>(int, int, std::vector<double>&)> rec =
        [&](int i, int p, std::vector<double>& opp) -> std::vector<double> {
        Node& nd = nodes_[i];
        if (nd.kind == Node::StreetEnd) return br_leaf[i][p];
        if (nd.kind != Node::Decision) {
            std::vector<double> dummy(kCombos, 0.0);
            return walk(i, p, 0, /*update=*/false, dummy, opp);
        }
        const int A = static_cast<int>(nd.acts.size());
        std::vector<double> cfv(kCombos, 0.0);
        if (nd.player == p) {
            bool first = true;
            for (int k = 0; k < A; ++k) {
                auto cv = rec(nd.child[k], p, opp);
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
                if (valid_[x] && opp[x] > 0.0)
                    child_opp[x] =
                        opp[x] * policy_row(nd, x, /*average=*/true)[k];
            auto cv = rec(nd.child[k], p, child_opp);
            for (int x = 0; x < kCombos; ++x) cfv[x] += cv[x];
        }
        return cfv;
    };

    double total = 0.0, joint = 0.0;
    std::vector<double> mass;
    compat_mass(root_.r1, valid_, mass);
    for (int i = 0; i < kCombos; ++i) joint += root_.r0[i] * mass[i];
    for (int p = 0; p < 2; ++p) {
        std::vector<double> opp = p == 0 ? root_.r1 : root_.r0;
        auto br = rec(0, p, opp);
        const auto& own = p == 0 ? root_.r0 : root_.r1;
        for (int x = 0; x < kCombos; ++x)
            if (valid_[x]) total += own[x] * br[x];
    }
    return joint > kTiny ? total / joint : 0.0;
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
            // mass floor: per-combo values with negligible opponent mass are
            // numerically meaningless (tiny/tiny) — exclude from targets
            if (!valid_[x] || own[x] <= kTiny || m[x] <= 1e-6) continue;
            v[p][x] = cfv[x] / m[x];
            mask[p][x] = 1.0;
        }
    }
}

bool HunlSolver::sample_leaf(std::mt19937& rng, double eps, int explorer,
                             int* leaf_node, uint8_t* card, HunlPBS* beta) {
    if (nb_root_ == 0) return false;   // preflop t*-leaf = train_preflop,
                                       // a later stage (3-card sample)
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const auto& ct = ComboTable::get();
    // sample compatible (hero, villain) combos from the root ranges
    double ws = 0.0;
    for (int x = 0; x < kCombos; ++x) {
        if (root_.r0[x] <= 0.0) continue;
        // marginal weight: r0[x] * compatible r1 mass — cheap enough exactly
        ws += root_.r0[x];
    }
    int h0 = -1, h1 = -1;
    for (int tries = 0; tries < 1000 && h1 < 0; ++tries) {
        double r = u01(rng) * ws, acc = 0.0;
        for (int x = 0; x < kCombos; ++x) {
            acc += root_.r0[x];
            if (r <= acc) {
                h0 = x;
                break;
            }
        }
        if (h0 < 0) h0 = 0;
        // rejection-sample villain compatible with hero
        const int a0 = ct.cards[h0][0], b0 = ct.cards[h0][1];
        double vs = 0.0;
        for (int y = 0; y < kCombos; ++y) {
            if (root_.r1[y] <= 0.0) continue;
            const int a = ct.cards[y][0], b = ct.cards[y][1];
            if (a == a0 || a == b0 || b == a0 || b == b0) continue;
            vs += root_.r1[y];
        }
        if (vs <= 0.0) continue;
        double rv = u01(rng) * vs, av = 0.0;
        for (int y = 0; y < kCombos; ++y) {
            if (root_.r1[y] <= 0.0) continue;
            const int a = ct.cards[y][0], b = ct.cards[y][1];
            if (a == a0 || a == b0 || b == a0 || b == b0) continue;
            av += root_.r1[y];
            if (rv <= av) {
                h1 = y;
                break;
            }
        }
    }
    if (h1 < 0) return false;
    const int hand[2] = {h0, h1};

    int i = 0;
    while (true) {
        const Node& nd = nodes_[i];
        if (nd.kind == Node::Fold || nd.kind == Node::Showdown ||
            nd.kind == Node::AllinShowdown)
            return false;                       // episode ends in-subgame
        if (nd.kind == Node::StreetEnd) {
            // sample the next card avoiding board + both sampled hands
            bool dead[kCards] = {};
            for (int b = 0; b < nb_root_; ++b) dead[board_[b]] = true;
            for (int p = 0; p < 2; ++p) {
                dead[ct.cards[hand[p]][0]] = true;
                dead[ct.cards[hand[p]][1]] = true;
            }
            std::vector<uint8_t> avail;
            for (int c = 0; c < kCards; ++c)
                if (!dead[c]) avail.push_back(static_cast<uint8_t>(c));
            const uint8_t c = avail[rng() % avail.size()];
            *leaf_node = i;
            *card = c;
            reaches_to(i, /*average=*/true, beta->r0, beta->r1);
            mask_card(beta->r0, c);
            mask_card(beta->r1, c);
            // normalize (fallback uniform over live combos if zero mass)
            std::vector<uint8_t> v2;
            std::array<uint8_t, 5> nb = board_;
            nb[nb_root_] = c;
            board_valid(nb.data(), nb_root_ + 1, v2);
            normalize_range(beta->r0, v2);
            normalize_range(beta->r1, v2);
            return true;
        }
        const int A = static_cast<int>(nd.acts.size());
        const int me = nd.player;
        int k = 0;
        if (me == explorer && u01(rng) < eps) {
            k = static_cast<int>(rng() % A);
        } else {
            auto pol = policy_row(nd, hand[me], /*average=*/true);
            double r = u01(rng), acc = 0.0;
            for (int j = 0; j < A; ++j) {
                acc += pol[j];
                k = j;
                if (r <= acc) break;
            }
        }
        i = nd.child[k];
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
