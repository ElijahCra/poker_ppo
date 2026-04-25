#include "elo_league.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace poker_ppo {

// ═════════════════════════════════════════════════════════════════════════════
// Construction
// ═════════════════════════════════════════════════════════════════════════════

EloLeague::EloLeague(IPokerEnvironmentFactory& factory,
                     const BetConfig& bet_cfg,
                     int obs_dim,
                     int action_count,
                     int hidden_dim,
                     int num_layers,
                     BetHistoryConfig hist,
                     Config cfg,
                     torch::Device device)
    : factory_(factory),
      bet_cfg_(bet_cfg),
      cfg_(cfg),
      obs_dim_(obs_dim),
      action_count_(action_count),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      hist_(hist),
      device_(device) {}

// ═════════════════════════════════════════════════════════════════════════════
// Snapshotting
// ═════════════════════════════════════════════════════════════════════════════

ActorCritic EloLeague::clone_network(const ActorCritic& src) {
    // Construct a fresh ActorCritic with identical architecture, then deep-copy
    // every parameter and buffer tensor.  This decouples the snapshot from
    // any future updates to `src`.
    ActorCritic dst(obs_dim_, action_count_, hidden_dim_, num_layers_, hist_);
    dst->to(device_);

    torch::NoGradGuard ng;

    auto src_params = src->parameters();
    auto dst_params = dst->parameters();
    TORCH_CHECK(src_params.size() == dst_params.size(),
                "EloLeague::clone_network: parameter count mismatch");
    for (size_t i = 0; i < src_params.size(); ++i) {
        dst_params[i].copy_(src_params[i].detach().to(device_));
    }

    auto src_bufs = src->buffers();
    auto dst_bufs = dst->buffers();
    TORCH_CHECK(src_bufs.size() == dst_bufs.size(),
                "EloLeague::clone_network: buffer count mismatch");
    for (size_t i = 0; i < src_bufs.size(); ++i) {
        dst_bufs[i].copy_(src_bufs[i].detach().to(device_));
    }

    dst->eval();
    return dst;
}

int EloLeague::add_checkpoint(const ActorCritic& net,
                              int update_idx,
                              int global_step,
                              const std::string& label) {
    nets_.push_back(clone_network(net));
    ckpts_.push_back({update_idx, global_step, cfg_.initial_rating, label});
    anchored_.push_back(false);

    // Prune oldest non-anchored if we exceed the cap.
    while (static_cast<int>(ckpts_.size()) > cfg_.max_checkpoints) {
        int remove = -1;
        for (size_t i = 0; i < ckpts_.size(); ++i) {
            if (!anchored_[i]) { remove = static_cast<int>(i); break; }
        }
        if (remove < 0) break;  // everything is anchored
        nets_.erase(nets_.begin() + remove);
        ckpts_.erase(ckpts_.begin() + remove);
        anchored_.erase(anchored_.begin() + remove);
    }

    return static_cast<int>(ckpts_.size()) - 1;
}

void EloLeague::set_anchor(int idx, float rating) {
    if (idx < 0 || idx >= static_cast<int>(ckpts_.size())) return;
    ckpts_[idx].rating = rating;
    anchored_[idx]     = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Match play
// ═════════════════════════════════════════════════════════════════════════════

EloLeague::MatchResult
EloLeague::play_match_internal(ActorCritic& a, ActorCritic& b) {
    const int P       = cfg_.num_parallel_envs;
    const int target  = cfg_.num_hands_per_match;

    if (P <= 0 || target <= 0) return {};

    // ── set up parallel environments ────────────────────────────────────
    std::vector<std::unique_ptr<IPokerEnvironment>> envs;
    envs.reserve(P);
    for (int i = 0; i < P; ++i) envs.push_back(factory_.create(bet_cfg_));

    // Alternate seat assignment across envs so A plays seat 0 in half of them
    // and seat 1 in the other half.  We also flip the assignment on every
    // hand completion for extra bias cancellation.
    std::vector<int> a_seat(P);
    for (int i = 0; i < P; ++i) a_seat[i] = i % 2;

    // Pull initial observation + legal mask from each env.
    std::vector<torch::Tensor> obs(P);
    std::vector<torch::Tensor> masks(P);
    for (int i = 0; i < P; ++i) {
        auto r   = envs[i]->reset();
        obs[i]   = r.observation;
        masks[i] = r.legal_action_mask;
    }

    torch::NoGradGuard ng;
    a->eval();
    b->eval();

    double total_reward_a = 0.0;
    int    wins_a = 0, losses_a = 0, ties_a = 0;
    int    hands_done = 0;
    std::vector<int64_t> action_counts_a(action_count_, 0);

    // Temporary batched tensors reused each step.
    auto batched_obs_buf  = torch::zeros({P, obs_dim_},      device_);
    auto batched_mask_buf = torch::zeros({P, action_count_}, device_);

    while (hands_done < target) {
        // Partition envs by which network must act this step.
        std::vector<int> a_idxs, b_idxs;
        for (int i = 0; i < P; ++i) {
            int acting = envs[i]->current_player();
            (acting == a_seat[i] ? a_idxs : b_idxs).push_back(i);
        }

        std::vector<int> actions(P, 0);

        auto run_batch = [&](ActorCritic& net, const std::vector<int>& idxs) {
            if (idxs.empty()) return;
            const int n = static_cast<int>(idxs.size());
            auto bo = batched_obs_buf.narrow(0, 0, n);
            auto bm = batched_mask_buf.narrow(0, 0, n);
            for (int j = 0; j < n; ++j) {
                bo[j] = obs[idxs[j]].to(device_);
                bm[j] = masks[idxs[j]].to(device_);
            }
            auto ar = net->get_action(bo, bm);
            auto ac = ar.action.to(torch::kCPU);
            for (int j = 0; j < n; ++j) {
                actions[idxs[j]] = static_cast<int>(ac[j].item<int64_t>());
            }
        };

        run_batch(a, a_idxs);
        run_batch(b, b_idxs);

        // Tally A's chosen actions for the histogram diagnostic.
        for (int idx : a_idxs) {
            const int act = actions[idx];
            if (act >= 0 && act < action_count_) ++action_counts_a[act];
        }

        // Step each env with its selected action.
        for (int i = 0; i < P; ++i) {
            if (hands_done >= target) break;

            auto r = envs[i]->step(actions[i]);

            if (r.done) {
                // The env returns reward from player 1's (seat 0's) perspective.
                // Flip to A's perspective based on which seat A played.
                const float raw      = r.reward;
                const float a_reward = (a_seat[i] == 0) ? raw : -raw;

                total_reward_a += a_reward;
                if      (a_reward > 0.0f) ++wins_a;
                else if (a_reward < 0.0f) ++losses_a;
                else                       ++ties_a;
                ++hands_done;

                // Reset and alternate seat for this env.
                a_seat[i] = 1 - a_seat[i];
                auto rr  = envs[i]->reset();
                obs[i]   = rr.observation;
                masks[i] = rr.legal_action_mask;
            } else {
                obs[i]   = r.observation;
                masks[i] = r.legal_action_mask;
            }
        }
    }

    MatchResult mr;
    mr.num_hands       = hands_done;
    mr.avg_reward_a    = static_cast<float>(total_reward_a / hands_done);
    mr.win_rate_a      = (static_cast<float>(wins_a) + 0.5f * ties_a) /
                         static_cast<float>(hands_done);
    mr.action_counts_a = std::move(action_counts_a);
    return mr;
}

EloLeague::MatchResult EloLeague::play_match(int i, int j) {
    TORCH_CHECK(i >= 0 && i < static_cast<int>(nets_.size()),
                "play_match: index i out of range");
    TORCH_CHECK(j >= 0 && j < static_cast<int>(nets_.size()),
                "play_match: index j out of range");
    return play_match_internal(nets_[i], nets_[j]);
}

// ═════════════════════════════════════════════════════════════════════════════
// Elo update
// ═════════════════════════════════════════════════════════════════════════════

float EloLeague::reward_to_score(float avg_reward_a) const {
    // Logistic squash on chip-EV.  avg_reward_a is in the env's *scaled* reward
    // units (mbb / reward_norm).  This is what Elo *should* see for poker —
    // hand-fraction wins ignore the size of pots won/lost, so a tight player
    // can win <50% of hands while crushing on bb/hand.
    const float s = cfg_.score_scale > 0.0f ? cfg_.score_scale : 1.0f;
    return 1.0f / (1.0f + std::exp(-avg_reward_a / s));
}

void EloLeague::update_ratings(int i, int j, float score_i) {
    const float ra = ckpts_[i].rating;
    const float rb = ckpts_[j].rating;

    // Standard logistic Elo expected score.
    const float expected_i = 1.0f / (1.0f + std::pow(10.0f, (rb - ra) / 400.0f));
    const float expected_j = 1.0f - expected_i;
    const float score_j    = 1.0f - score_i;

    const float k = cfg_.k_factor;
    if (!anchored_[i]) ckpts_[i].rating += k * (score_i - expected_i);
    if (!anchored_[j]) ckpts_[j].rating += k * (score_j - expected_j);
}

// ═════════════════════════════════════════════════════════════════════════════
// Tournaments
// ═════════════════════════════════════════════════════════════════════════════

void EloLeague::run_tournament() {
    const int N = static_cast<int>(ckpts_.size());
    if (N < 2) return;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            auto mr = play_match_internal(nets_[i], nets_[j]);
            update_ratings(i, j, reward_to_score(mr.avg_reward_a));
        }
    }
}

void EloLeague::evaluate_latest() {
    const int N = static_cast<int>(ckpts_.size());
    if (N < 2) return;
    const int latest = N - 1;

    for (int i = 0; i < latest; ++i) {
        auto mr = play_match_internal(nets_[latest], nets_[i]);
        update_ratings(latest, i, reward_to_score(mr.avg_reward_a));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Display
// ═════════════════════════════════════════════════════════════════════════════

void EloLeague::print_standings(std::ostream& os) const {
    std::vector<int> order(ckpts_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return ckpts_[a].rating > ckpts_[b].rating; });

    os << "────────────────────────────────────────────────────────────\n";
    os << "  Rank   Update       Step        Rating   Label\n";
    os << "────────────────────────────────────────────────────────────\n";
    for (size_t r = 0; r < order.size(); ++r) {
        const int k = order[r];
        os << std::setw(6)  << (r + 1)
           << std::setw(9)  << ckpts_[k].update_idx
           << std::setw(12) << ckpts_[k].global_step
           << std::setw(12) << std::fixed << std::setprecision(1)
                            << ckpts_[k].rating
           << "   " << ckpts_[k].label;
        if (anchored_[k]) os << "  [anchor]";
        os << "\n";
    }
    os << "────────────────────────────────────────────────────────────\n";
}

void EloLeague::print_standings() const { print_standings(std::cout); }

}  // namespace poker_ppo
