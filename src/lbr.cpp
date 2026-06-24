#include "lbr.h"

#include "poker_env.h"
#include "Utility/RangeEquity.hpp"

#include <array>
#include <chrono>
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
// the obs scratch fed to the target net. Combos colliding with dead cards
// (LBR hole + board) are pruned to weight 0 and skipped.
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
        for (size_t i = 0; i < combos.size(); ++i) {
            if (dead[combos[i][0]] || dead[combos[i][1]]) weights[i] = 0.0;
        }
        renormalize();
    }

    void renormalize() {
        double s = 0.0;
        for (double w : weights) s += w;
        if (s <= 0.0) {  // degenerate (numerical) — fall back to uniform
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

}  // namespace

LBREvaluator::Result LBREvaluator::evaluate(ActorCritic& target) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    target->eval();
    torch::NoGradGuard ng;

    const int D = env_->obs_dim();
    const int A = bet_cfg_.action_count();
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

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

        while (!env_->is_terminal()) {
            const int cur = env_->current_player();

            // Prune the belief against any newly-revealed board cards.
            mark_dead(dead, *env_, lbr_seat);
            belief.prune(dead);

            if (cur != lbr_seat) {
                // ── Target (villain) acts; observe + Bayes-update belief ──
                auto mask = env_->legal_action_mask();              // [A] cpu
                auto real_obs = env_->observation().unsqueeze(0).to(device_);
                auto mask_dev = mask.unsqueeze(0).to(device_);
                auto ar = target->get_action(real_obs, mask_dev);
                const int a = static_cast<int>(ar.action.item<int64_t>());

                // Active (nonzero-weight) combos only.
                std::vector<int> active;
                active.reserve(belief.combos.size());
                for (size_t i = 0; i < belief.combos.size(); ++i)
                    if (belief.weights[i] > 0.0) active.push_back(
                        static_cast<int>(i));

                if (!active.empty()) {
                    auto obs_batch = torch::empty(
                        {static_cast<long>(active.size()), D}, f_cpu);
                    float* base = obs_batch.data_ptr<float>();
                    for (size_t r = 0; r < active.size(); ++r) {
                        const auto& c = belief.combos[active[r]];
                        env_->observation_for_hole(c[0], c[1], base + r * D);
                    }
                    auto obs_dev  = obs_batch.to(device_);
                    auto mb_dev   = mask.unsqueeze(0)
                                        .expand({static_cast<long>(active.size()), A})
                                        .to(device_);
                    auto logp = target->masked_log_probs(obs_dev, mb_dev);
                    auto pa   = logp.select(1, a).exp().to(torch::kCPU).contiguous();
                    auto pacc = pa.accessor<float, 1>();
                    for (size_t r = 0; r < active.size(); ++r)
                        belief.weights[active[r]] *=
                            static_cast<double>(pacc[static_cast<long>(r)]);
                    belief.renormalize();
                }

                env_->step(a);
            } else {
                // ── LBR acts: call iff equity beats pot odds, else fold ──
                const auto h = env_->hole_cards(lbr_seat);
                const int  nb = env_->community_count();
                uint8_t board[5];
                for (int i = 0; i < nb; ++i)
                    board[i] = static_cast<uint8_t>(env_->community_card(i));

                const int c  = env_->amount_to_call();
                if (c <= 0) {
                    env_->step(1);  // free check — never fold
                } else {
                    const double e = Game::hand_vs_range_equity(
                        static_cast<uint8_t>(h[0]), static_cast<uint8_t>(h[1]),
                        board, nb, belief.combos, belief.weights,
                        rng_, cfg_.equity_mc_samples);
                    const double pot_after = static_cast<double>(env_->pot() + c);
                    // ΔEV(call vs fold) = e·pot_after − c.
                    const bool call = e * pot_after > static_cast<double>(c);
                    env_->step(call ? 1 : 0);
                }
            }
        }

        const double u = static_cast<double>(env_->terminal_utility(lbr_seat));
        total_mbb += u;
        if      (u > 0.0) ++wins;
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
