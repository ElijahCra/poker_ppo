#include "hunl_play.h"

#include <algorithm>
#include <cstdio>

namespace rebel_hunl {

using poker_ppo::PokerEnvironment;

// ── CheckCallTarget ───────────────────────────────────────────────────────

torch::Tensor CheckCallTarget::probs_for_holes(
    PokerEnvironment& env, const torch::Tensor& mask,
    const std::vector<std::array<uint8_t, 2>>& holes) {
    (void)env;
    const long n = static_cast<long>(holes.size());
    const long A = mask.size(0);
    auto m = mask.accessor<float, 1>();
    const int a = m[1] > 0.5f ? 1 : 0;
    auto P = torch::zeros({n, A}, torch::kFloat);
    P.index_put_({torch::indexing::Slice(), a}, 1.0f);
    return P;
}

int CheckCallTarget::act(PokerEnvironment& env, const torch::Tensor& mask) {
    (void)env;
    auto m = mask.accessor<float, 1>();
    return m[1] > 0.5f ? 1 : 0;
}

// ── RebelTarget ───────────────────────────────────────────────────────────

RebelTarget::RebelTarget(HunlValueNet net, double stack,
                         torch::Device device, RebelPlayConfig cfg,
                         poker_ppo::ILBRTarget* blueprint)
    : net_(net), stack_(stack), device_(device), cfg_(std::move(cfg)),
      blueprint_(blueprint),
      oracle_(std::make_unique<HunlNetOracle>(net_, stack_, device_)),
      rng_(static_cast<unsigned>(cfg_.seed ? cfg_.seed
                                           : std::random_device{}())) {
    pbs_.r0.assign(kCombos, 1.0 / kCombos);
    pbs_.r1.assign(kCombos, 1.0 / kCombos);
}

void RebelTarget::on_hand_start(PokerEnvironment& env) {
    pbs_.r0.assign(kCombos, 1.0 / kCombos);
    pbs_.r1.assign(kCombos, 1.0 / kCombos);
    board_seen_ = 0;
    clear_solves();
    cache_round_ = -1;
    seat_ = -1;
    have_alt_ = false;
    sync_public(env);
}

void RebelTarget::clear_solves() {
    // The GAME-ROOT preflop solve (empty action log) is identical every
    // hand: uniform PBS, no board, blind pot, fixed pf flop sample, and
    // CFR is deterministic — so it is solved once per agent lifetime and
    // survives both the per-hand and per-street cache clears. Bit-equal
    // policies to re-solving, at ~one preflop solve per SHARD per gate
    // instead of per hand.
    if (cfg_.preflop_solve) {
        cache_.erase(std::remove_if(cache_.begin(), cache_.end(),
                                    [](const Solve& s) {
                                        return !s.log_at_root.empty();
                                    }),
                     cache_.end());
    } else {
        cache_.clear();
    }
}

void RebelTarget::alt_from_turn_leaf(PokerEnvironment& env) {
    have_alt_ = false;
    const auto& log = env.action_log();
    for (auto& sv : cache_) {   // cache_ still holds the TURN solves
        if (sv.log_at_root.size() > log.size() ||
            !std::equal(sv.log_at_root.begin(), sv.log_at_root.end(),
                        log.begin()))
            continue;
        HunlSolver& s = *sv.solver;
        int node = 0;
        bool ok = true;
        for (size_t i = sv.log_at_root.size(); i < log.size() && ok; ++i) {
            const auto& nd = s.nodes()[static_cast<size_t>(node)];
            if (nd.kind != HunlSolver::Node::Decision) {
                ok = false;
                break;
            }
            int k = -1;
            for (size_t j = 0; j < nd.acts.size(); ++j)
                if (nd.acts[j] == log[i]) {
                    k = static_cast<int>(j);
                    break;
                }
            if (k < 0) ok = false;
            else       node = nd.child[static_cast<size_t>(k)];
        }
        if (!ok ||
            s.nodes()[static_cast<size_t>(node)].kind !=
                HunlSolver::Node::StreetEnd)
            continue;
        // CFR-AVG beliefs at the leaf, masked by the dealt river card —
        // exactly the PBS the turn solve priced this continuation at
        HunlPBS lb;
        s.beliefs_at(node, lb.r0, lb.r1);
        uint8_t b5[5];
        for (int i = 0; i < 5; ++i)
            b5[i] = static_cast<uint8_t>(env.community_card(i));
        const auto& ct = ComboTable::get();
        for (int i = 0; i < kCombos; ++i)
            if (ct.cards[i][0] == b5[4] || ct.cards[i][1] == b5[4])
                lb.r0[i] = lb.r1[i] = 0.0;
        std::array<std::vector<double>, 2> out;
        oracle_->value(env, b5, 5, lb, out);
        alt_ = std::move(out);
        have_alt_ = true;
        return;
    }
}

void RebelTarget::sync_public(PokerEnvironment& env) {
    const int nb = env.community_count();
    if (nb < board_seen_) {   // defensive: hand restarted without the hook
        on_hand_start(env);
        return;
    }
    if (nb > board_seen_) {
        const auto& ct = ComboTable::get();
        for (int b = board_seen_; b < nb; ++b) {
            const auto c = static_cast<uint8_t>(env.community_card(b));
            for (int i = 0; i < kCombos; ++i)
                if (ct.cards[i][0] == c || ct.cards[i][1] == c)
                    pbs_.r0[i] = pbs_.r1[i] = 0.0;
        }
        for (auto* r : {&pbs_.r0, &pbs_.r1}) {
            double s = 0.0;
            for (double w : *r) s += w;
            if (s > 0.0)
                for (double& w : *r) w /= s;
        }
        board_seen_ = nb;
    }
}

std::pair<HunlSolver*, int> RebelTarget::solve_at(PokerEnvironment& env) {
    // cache management is per street (public state changed = new trees)
    if (env.round() != cache_round_) {
        // capture the opponent's river alternatives from the turn solve's
        // leaf BEFORE the turn cache drops
        if (cfg_.gadget && env.round() == 3 && cache_round_ == 2)
            alt_from_turn_leaf(env);
        clear_solves();
        cache_round_ = env.round();
    }
    const auto& log = env.action_log();
    for (auto& sv : cache_) {
        if (sv.log_at_root.size() > log.size() ||
            !std::equal(sv.log_at_root.begin(), sv.log_at_root.end(),
                        log.begin()))
            continue;
        HunlSolver& s = *sv.solver;
        int node = 0;
        bool ok = true;
        for (size_t i = sv.log_at_root.size(); i < log.size() && ok; ++i) {
            const auto& nd = s.nodes()[static_cast<size_t>(node)];
            if (nd.kind != HunlSolver::Node::Decision) {
                ok = false;
                break;
            }
            int k = -1;
            for (size_t j = 0; j < nd.acts.size(); ++j)
                if (nd.acts[j] == log[i]) {
                    k = static_cast<int>(j);
                    break;
                }
            if (k < 0) ok = false;
            else       node = nd.child[static_cast<size_t>(k)];
        }
        if (ok &&
            s.nodes()[static_cast<size_t>(node)].kind ==
                HunlSolver::Node::Decision)
            return {&s, node};
    }
    // fresh solve rooted at the current node with the tracked PBS (the
    // solver masks/normalizes its own copy against the env's board)
    const int round = env.round();
    const bool river = round >= 3;
    auto solver = std::make_unique<HunlSolver>(
        env, pbs_, river ? nullptr : oracle_.get(),
        river ? cfg_.actions_river
              : round <= 1 ? cfg_.actions_flop : cfg_.actions);
    solver->refresh_every =
        round <= 1 ? cfg_.refresh_flop : cfg_.refresh_every;
    // pcfr_bench: PCFR+ wins 1.5-3.5x on the sparse net-leaf streets but
    // LOSES ~2x on river trees — river stays CFR+ (also keeps the GPU
    // river solver's CPU-equivalence meaningful)
    solver->pcfr = cfg_.pcfr && !river;
    solver->pf_samples = cfg_.pf_samples;
    if (cfg_.seed)   // sampled preflop-leaf flops vary per shard
        solver->pf_seed = cfg_.seed * 6364136223846793005ull + 1442695040888963407ull;
    if (river && cfg_.gadget && seat_ >= 0) {
        // a mid-street fresh solve (off-tree action) inherits alternatives
        // from the deepest in-tree node of a previous river solve — the
        // opponent's own deviation cannot raise its entitlement
        if (!cache_.empty()) {
            HunlSolver& prev = *cache_.back().solver;
            const auto& rlog = cache_.back().log_at_root;
            if (rlog.size() <= log.size() &&
                std::equal(rlog.begin(), rlog.end(), log.begin())) {
                int node = 0;
                for (size_t i = rlog.size(); i < log.size(); ++i) {
                    const auto& nd =
                        prev.nodes()[static_cast<size_t>(node)];
                    if (nd.kind != HunlSolver::Node::Decision) break;
                    int k = -1;
                    for (size_t j = 0; j < nd.acts.size(); ++j)
                        if (nd.acts[j] == log[i]) {
                            k = static_cast<int>(j);
                            break;
                        }
                    if (k < 0) break;
                    node = nd.child[static_cast<size_t>(k)];
                }
                std::array<std::vector<double>, 2> v, m;
                prev.values_at(node, v, m);
                alt_ = std::move(v);
                have_alt_ = true;
            }
        }
        if (have_alt_)
            solver->enable_gadget(1 - seat_, alt_[static_cast<size_t>(
                                      1 - seat_)], cfg_.gadget_mix);
    }
    const int T = river ? cfg_.t_river
                : round == 2 ? cfg_.t_turn
                : round == 1 ? cfg_.t_flop : cfg_.t_preflop;
    bool solved = false;
    if (cfg_.gpu_turn && round == 2 && !cfg_.gadget)
        solved = gpu_turn_solve(*solver, T);
    if (!solved)
        for (int t = 1; t <= T; ++t) solver->iterate(t);
    cache_.push_back(Solve{log, std::move(solver)});
    return {cache_.back().solver.get(), 0};
}

bool RebelTarget::gpu_turn_solve(HunlSolver& s, int T) {
    if (!torch::cuda::is_available() || s.board_count() != 4) return false;
    try {
        TurnSpec sp;
        for (int i = 0; i < 4; ++i)
            sp.board[static_cast<size_t>(i)] = s.board()[static_cast<size_t>(i)];
        sp.r0 = pbs_.r0;
        sp.r1 = pbs_.r1;
        const auto& nodes = s.nodes();
        sp.node_contrib0.resize(nodes.size());
        sp.node_contrib1.resize(nodes.size());
        for (size_t m = 0; m < nodes.size(); ++m) {
            sp.node_contrib0[m] = nodes[m].contrib[0];
            sp.node_contrib1[m] = nodes[m].contrib[1];
        }
        auto shape = TreeShape::from(s);
        torch::Device dev(torch::kCUDA);
        std::vector<TurnSpec> specs{sp};
        BatchTurnSolver g(shape, specs, dev, torch::kFloat, cfg_.pcfr);
        HunlNetOracle oracle(net_, stack_, dev);
        std::vector<HunlNetOracle*> ora{&oracle};
        // device featurization by default; CPU-oracle fallback keeps
        // threads=1 (gates parallelize across shards, not within)
        const bool dev_feat = !std::getenv("REBEL_NO_DEV_FEAT");
        g.solve(T, cfg_.refresh_every, [&](int) {
            if (dev_feat)
                g.refresh_leaves_device(net_, stack_);
            else
                refresh_turn_leaves(g, specs, ora, /*threads=*/1, &gpu_ws_);
        });
        for (size_t m = 0; m < nodes.size(); ++m) {
            if (nodes[m].kind != HunlSolver::Node::Decision) continue;
            const int A = static_cast<int>(nodes[m].acts.size());
            auto cum = g.cum_state(static_cast<int>(m))
                           .select(0, 0)
                           .to(torch::kCPU)
                           .to(torch::kDouble)
                           .contiguous();   // [A, n]
            const double* cp = cum.data_ptr<double>();
            std::vector<double> row(static_cast<size_t>(A) * kCombos);
            for (int x = 0; x < kCombos; ++x)
                for (int k = 0; k < A; ++k)
                    row[static_cast<size_t>(x) * A + k] =
                        cp[static_cast<size_t>(k) * kCombos + x];
            s.set_cum_strat(static_cast<int>(m), row);
        }
        return true;
    } catch (const std::exception& e) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "[gpu-turn] play-time solve failed (%s) — CPU "
                         "fallback\n", e.what());
            warned = true;
        }
        return false;
    }
}

void RebelTarget::apply_range_update(int seat,
                                     const std::vector<double>& col) {
    std::vector<double>& r = seat == 0 ? pbs_.r0 : pbs_.r1;
    double s = 0.0;
    for (int i = 0; i < kCombos; ++i) s += r[i] * col[i];
    if (s <= 1e-300) return;   // model gave the action zero mass: keep range
    for (int i = 0; i < kCombos; ++i) r[i] = r[i] * col[i] / s;
}

torch::Tensor RebelTarget::probs_for_holes(
    PokerEnvironment& env, const torch::Tensor& mask,
    const std::vector<std::array<uint8_t, 2>>& holes) {
    seat_ = env.current_player();   // policy queries are about OUR node
    if (env.round() < solve_from() && blueprint_)
        return blueprint_->probs_for_holes(env, mask, holes);
    if (env.round() < solve_from()) {
        CheckCallTarget stub;
        return stub.probs_for_holes(env, mask, holes);
    }
    auto [s, node] = solve_at(env);
    const auto& nd = s->nodes()[static_cast<size_t>(node)];
    const auto& ct = ComboTable::get();
    const long n = static_cast<long>(holes.size());
    const long A = mask.size(0);
    auto P = torch::zeros({n, A}, torch::kFloat);
    auto acc = P.accessor<float, 2>();
    for (long r = 0; r < n; ++r) {
        const int combo =
            ct.id[holes[static_cast<size_t>(r)][0]]
                 [holes[static_cast<size_t>(r)][1]];
        auto pol = s->avg_policy(node, combo);
        for (size_t k = 0; k < nd.acts.size(); ++k)
            acc[r][nd.acts[k]] = static_cast<float>(pol[k]);
    }
    return P;
}

int RebelTarget::act(PokerEnvironment& env, const torch::Tensor& mask) {
    sync_public(env);
    seat_ = env.current_player();
    if (env.round() < solve_from()) {
        if (blueprint_) return blueprint_->act(env, mask);
        CheckCallTarget stub;
        return stub.act(env, mask);
    }
    auto [s, node] = solve_at(env);
    const auto& nd = s->nodes()[static_cast<size_t>(node)];
    const auto h = env.hole_cards(env.current_player());
    const int combo = ComboTable::get().id[h[0]][h[1]];
    auto pol = s->avg_policy(node, combo);
    double sum = 0.0;
    for (double p : pol) sum += p;
    if (sum <= 0.0) {   // degenerate: check/call
        auto m = mask.accessor<float, 1>();
        return m[1] > 0.5f ? 1 : 0;
    }
    std::uniform_real_distribution<double> u(0.0, sum);
    double x = u(rng_);
    for (size_t k = 0; k < pol.size(); ++k) {
        x -= pol[k];
        if (x <= 0.0) return nd.acts[k];
    }
    return nd.acts.back();
}

void RebelTarget::note_action(PokerEnvironment& env, int action) {
    sync_public(env);
    const int seat = env.current_player();
    if (env.round() < solve_from()) {
        if (!blueprint_) return;   // check/call stub: uninformative update
        // blueprint model for the acting seat's range (all live combos)
        const std::vector<double>& r = seat == 0 ? pbs_.r0 : pbs_.r1;
        const auto& ct = ComboTable::get();
        std::vector<std::array<uint8_t, 2>> holes;
        std::vector<int> idx;
        holes.reserve(kCombos);
        idx.reserve(kCombos);
        for (int i = 0; i < kCombos; ++i) {
            if (r[i] <= 0.0) continue;
            holes.push_back(ct.cards[i]);
            idx.push_back(i);
        }
        if (holes.empty()) return;
        auto P = blueprint_->probs_for_holes(env, env.legal_action_mask(),
                                             holes);
        auto acc = P.accessor<float, 2>();
        std::vector<double> col(kCombos, 0.0);
        for (size_t k = 0; k < idx.size(); ++k)
            col[static_cast<size_t>(idx[k])] =
                static_cast<double>(acc[static_cast<long>(k)][action]);
        apply_range_update(seat, col);
        return;
    }
    auto [s, node] = solve_at(env);
    const auto& nd = s->nodes()[static_cast<size_t>(node)];
    int k = -1;
    for (size_t j = 0; j < nd.acts.size(); ++j)
        if (nd.acts[j] == action) {
            k = static_cast<int>(j);
            break;
        }
    if (k < 0) {
        // Off-tree raise: pseudo-harmonic action mapping (Ganzfried &
        // Sandholm 2013) onto the two nearest in-tree sizes A ≤ x ≤ B
        // (pot fractions; call counts as size 0). Skipping the update —
        // the old behavior — leaks: the raiser's range stays
        // un-conditioned on the raise, a blind spot an exploiter can
        // aim off-tree sizes at. The observation likelihood becomes
        // w·π(A|combo) + (1−w)·π(B|combo), w from the harmonic map.
        if (action < 2) return;   // non-raise: nothing to map
        static const double kFrac[13] = {0,   0,   0.25, 0.33, 0.5,
                                         0.66, 0.75, 1.0, 1.25, 1.5,
                                         2.0, 2.5, 3.0};
        const double c   = static_cast<double>(env.amount_to_call());
        const double pot = static_cast<double>(env.pot());
        auto frac_of = [&](int a) {
            if (a == 1) return 0.0;
            if (a >= 2 && a <= 12) return kFrac[a];
            // all-in: raise-by as a fraction of the called pot
            return (static_cast<double>(env.stack(seat)) - c) /
                   std::max(1.0, pot + c);
        };
        const double x = frac_of(action);
        int ka = -1, kb = -1;
        double fa = 0.0, fb = 0.0;
        for (size_t j = 0; j < nd.acts.size(); ++j) {
            const int a = nd.acts[j];
            if (a == 0) continue;   // fold is not a size
            const double f = frac_of(a);
            if (f <= x && (ka < 0 || f > fa)) {
                ka = static_cast<int>(j);
                fa = f;
            }
            if (f >= x && (kb < 0 || f < fb)) {
                kb = static_cast<int>(j);
                fb = f;
            }
        }
        if (ka < 0 && kb < 0) return;
        if (ka < 0) {   // below every in-tree size: full weight above
            ka = kb;
            kb = -1;
        }
        double w = 1.0;   // kb<0 (above every size): clamp to largest
        if (kb >= 0 && kb != ka && fb > fa)
            w = ((fb - x) * (1.0 + fa)) / ((fb - fa) * (1.0 + x));
        w = std::min(1.0, std::max(0.0, w));
        std::vector<double> col(kCombos, 0.0);
        for (int i = 0; i < kCombos; ++i) {
            const auto pol = s->avg_policy(node, i);
            double p = w * pol[static_cast<size_t>(ka)];
            if (kb >= 0 && kb != ka)
                p += (1.0 - w) * pol[static_cast<size_t>(kb)];
            col[static_cast<size_t>(i)] = p;
        }
        apply_range_update(seat, col);
        return;
    }
    std::vector<double> col(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i)
        col[static_cast<size_t>(i)] =
            s->avg_policy(node, i)[static_cast<size_t>(k)];
    apply_range_update(seat, col);
}

}  // namespace rebel_hunl
