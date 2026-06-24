#include "lbr.h"

#include "poker_env.h"
#include "Utility/RangeEquity.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace poker_ppo {

LBREvaluator::LBREvaluator(IPokerEnvironmentFactory& factory,
                           const BetConfig&          bet_cfg,
                           LBRConfig                 cfg,
                           torch::Device             device)
    : factory_(factory), bet_cfg_(bet_cfg), cfg_(cfg), device_(device),
      rng_(cfg.seed ? cfg.seed : std::random_device{}())
{
    env_owned_ = factory_.create(bet_cfg_);
    env_       = dynamic_cast<PokerEnvironment*>(env_owned_.get());
    TORCH_CHECK(env_ != nullptr,
                "LBREvaluator: factory must produce a PokerEnvironment");
}

namespace {

// Active belief over the villain's holding: parallel combos / weights, plus
// pruning of combos that collide with dead cards (LBR hole + board).
struct Belief {
    std::vector<std::array<uint8_t, 2>> combos;
    std::vector<double>                 weights;

    void init(const bool* dead) {
        combos.clear();
        weights.clear();
        for (int a = 0; a < 52; ++a) {
            if (dead[a]) continue;
            for (int b = a + 1; b < 52; ++b) {
                if (dead[b]) continue;
                combos.push_back({static_cast<uint8_t>(a),
                                  static_cast<uint8_t>(b)});
                weights.push_back(1.0);
            }
        }
        renormalize();
    }

    void prune(const bool* dead) {
        for (size_t i = 0; i < combos.size(); ++i)
            if (dead[combos[i][0]] || dead[combos[i][1]]) weights[i] = 0.0;
        renormalize();
    }

    void renormalize() {
        double s = 0.0;
        for (double w : weights) s += w;
        if (s <= 0.0) {
            for (double& w : weights) w = 1.0;
            s = static_cast<double>(weights.size());
        }
        const double inv = 1.0 / s;
        for (double& w : weights) w *= inv;
    }
};

void mark_dead(bool* dead, const PokerEnvironment& env, int lbr_seat) {
    std::fill(dead, dead + 52, false);
    const auto h = env.hole_cards(lbr_seat);
    dead[h[0]] = dead[h[1]] = true;
    const int nb = env.community_count();
    for (int i = 0; i < nb; ++i) dead[env.community_card(i)] = true;
}

// Legal raise candidates to price: nearest legal raises to 0.5× / 1× pot,
// plus all-in. A small set keeps cost bounded and matches LBR's fixed bet
// menu (Lisý & Bowling); two sizes already expose heavy exploitability.
std::vector<int> candidate_raises(const PokerEnvironment& env,
                                  const torch::Tensor& mask) {
    const auto& fr = env.game_config().pot_fractions;
    auto m = mask.accessor<float, 1>();
    std::vector<int> out;
    auto add_nearest = [&](double target) {
        int best = -1; double bd = 1e18;
        for (size_t j = 0; j < fr.size(); ++j) {
            const int idx = 2 + static_cast<int>(j);
            if (idx < mask.size(0) && m[idx] > 0.5f) {
                const double d = std::abs(fr[j] - target);
                if (d < bd) { bd = d; best = idx; }
            }
        }
        if (best >= 0 && std::find(out.begin(), out.end(), best) == out.end())
            out.push_back(best);
    };
    add_nearest(0.5);
    add_nearest(1.0);
    const int allin = 2 + static_cast<int>(fr.size());
    if (allin < mask.size(0) && m[allin] > 0.5f &&
        std::find(out.begin(), out.end(), allin) == out.end())
        out.push_back(allin);
    return out;
}

}  // namespace

LBREvaluator::Result LBREvaluator::evaluate(ActorCritic& target) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    target->eval();
    torch::NoGradGuard ng;

    const int D = env_->obs_dim();
    const int A = bet_cfg_.action_count();
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Target policy probs at the CURRENT node for each active combo, holding
    // it counterfactually: [n_active, A] on CPU. mask is the public legal
    // mask (card-independent). Used for the Bayes update and for reading the
    // villain's fold/call response to a candidate LBR raise.
    auto active_probs = [&](const Belief& b, const std::vector<int>& active,
                            const torch::Tensor& mask) -> torch::Tensor {
        const long n = static_cast<long>(active.size());
        auto obs = torch::empty({n, D}, f_cpu);
        float* base = obs.data_ptr<float>();
        for (long r = 0; r < n; ++r) {
            const auto& c = b.combos[active[r]];
            env_->observation_for_hole(c[0], c[1], base + r * D);
        }
        auto obs_dev = obs.to(device_);
        auto mb_dev  = mask.unsqueeze(0).expand({n, A}).to(device_);
        auto logp    = target->masked_log_probs(obs_dev, mb_dev);
        return logp.exp().to(torch::kCPU).contiguous();  // [n, A]
    };

    auto active_of = [](const Belief& b) {
        std::vector<int> active;
        active.reserve(b.combos.size());
        for (size_t i = 0; i < b.combos.size(); ++i)
            if (b.weights[i] > 0.0) active.push_back(static_cast<int>(i));
        return active;
    };

    double total_mbb = 0.0;
    int    wins = 0, ties = 0;
    const int big_blind = env_->game_config().big_blind;

    Belief belief;
    bool dead[52];

    for (int hand = 0; hand < cfg_.num_hands; ++hand) {
        const int lbr_seat = hand % 2;
        env_->reset();
        mark_dead(dead, *env_, lbr_seat);
        belief.init(dead);

        // Once LBR raises, it commits to calling down to showdown — matching
        // the checkdown EV model the raise was priced under. Without this,
        // re-optimising each street can fold the pot the raise inflated,
        // realising far less than the modelled value (raises overvalued).
        bool committed = false;

        while (!env_->is_terminal()) {
            const int cur = env_->current_player();
            mark_dead(dead, *env_, lbr_seat);
            belief.prune(dead);

            if (cur != lbr_seat) {
                // ── Target acts; observe its action, Bayes-update belief ──
                auto mask     = env_->legal_action_mask();
                auto real_obs = env_->observation().unsqueeze(0).to(device_);
                auto mask_dev = mask.unsqueeze(0).to(device_);
                const int a = static_cast<int>(
                    target->get_action(real_obs, mask_dev).action.item<int64_t>());

                auto active = active_of(belief);
                if (!active.empty()) {
                    auto P = active_probs(belief, active, mask);  // [n, A]
                    auto col = P.select(1, a).contiguous();
                    auto pa  = col.accessor<float, 1>();
                    for (size_t r = 0; r < active.size(); ++r)
                        belief.weights[active[r]] *=
                            static_cast<double>(pa[static_cast<long>(r)]);
                    belief.renormalize();
                }
                env_->step(a);
                continue;
            }

            // ── LBR acts ──────────────────────────────────────────────────
            auto mask = env_->legal_action_mask();
            // Committed (post-raise): call/check to showdown, never fold.
            if (committed) { env_->step(1); continue; }
            const auto h  = env_->hole_cards(lbr_seat);
            const int  nb = env_->community_count();
            uint8_t board[5];
            for (int i = 0; i < nb; ++i)
                board[i] = static_cast<uint8_t>(env_->community_card(i));

            const double pot0 = static_cast<double>(env_->pot());
            const int    c    = env_->amount_to_call();

            // Call/check baseline: ΔEV vs folding = e·(pot0+c) − c.
            const double e_call = Game::hand_vs_range_equity(
                static_cast<uint8_t>(h[0]), static_cast<uint8_t>(h[1]),
                board, nb, belief.combos, belief.weights,
                rng_, cfg_.equity_mc_samples);
            const double ev_call = e_call * (pot0 + c) - static_cast<double>(c);

            int    best_action = (c <= 0) ? 1 : (ev_call > 0.0 ? 1 : 0);
            double best_ev     = (c <= 0) ? ev_call : std::max(0.0, ev_call);

            // Raises are priced with a single-street checkdown model, which
            // is EXACT only on the river (no further cards or betting rounds
            // to mis-model). Earlier-street raises would need recursive
            // multi-street lookahead; approximating them produced badly
            // biased (too-low) bounds, so v2 raises only on the river — a
            // sound strict improvement over v1 (exact river value/bluff
            // raises), never a regression.
            if (cfg_.enable_raises && nb == 5) {
                auto active = active_of(belief);
                for (int a_r : candidate_raises(*env_, mask)) {
                    env_->push_state();
                    env_->step(a_r);
                    // Now at the villain's node facing the raise.
                    const double pot1 = static_cast<double>(env_->pot());
                    const double add_r = pot1 - pot0;          // LBR's chips in
                    const double vc    = static_cast<double>(env_->amount_to_call());
                    auto vmask = env_->legal_action_mask();

                    double p_fold = 0.0;
                    std::vector<double> call_w(active.size(), 0.0);
                    if (!active.empty()) {
                        auto P  = active_probs(belief, active, vmask);  // [n, A]
                        auto pf_t = P.select(1, 0).contiguous();
                        auto pf = pf_t.accessor<float, 1>();
                        for (size_t r = 0; r < active.size(); ++r) {
                            const double w  = belief.weights[active[r]];
                            const double pf_i = static_cast<double>(
                                pf[static_cast<long>(r)]);
                            p_fold   += w * pf_i;
                            call_w[r] = w * (1.0 - pf_i);  // reraise folded into call
                        }
                    }
                    env_->pop_state();

                    // Equity vs the call-range (combos reweighted by P(call)).
                    std::vector<std::array<uint8_t, 2>> cc(active.size());
                    for (size_t r = 0; r < active.size(); ++r)
                        cc[r] = belief.combos[active[r]];
                    const double e_raise = Game::hand_vs_range_equity(
                        static_cast<uint8_t>(h[0]), static_cast<uint8_t>(h[1]),
                        board, nb, cc, call_w, rng_, cfg_.equity_mc_samples);

                    // ΔEV vs fold: villain folds → win pot0; villain calls →
                    // equity on the final pot, minus LBR's raise outlay.
                    const double ev_raise =
                        p_fold * pot0
                        + (1.0 - p_fold) * (e_raise * (pot1 + vc) - add_r);

                    if (ev_raise > best_ev) {
                        best_ev = ev_raise;
                        best_action = a_r;
                    }
                }
            }

            if (best_action >= 2) committed = true;  // raised → call down after
            env_->step(best_action);
        }

        const double u = static_cast<double>(env_->terminal_utility(lbr_seat));
        total_mbb += u;
        if      (u > 0.0)  ++wins;
        else if (u == 0.0) ++ties;
    }

    using ms = std::chrono::duration<double, std::milli>;
    Result r;
    r.num_hands    = cfg_.num_hands;
    r.mbb_per_hand = total_mbb / std::max(1, cfg_.num_hands);
    r.bb_per_hand  = r.mbb_per_hand / std::max(1, big_blind);
    r.lbr_win_rate = (static_cast<double>(wins) + 0.5 * ties)
                     / std::max(1, cfg_.num_hands);
    r.wall_ms      = ms(clock::now() - t0).count();
    return r;
}

}  // namespace poker_ppo
