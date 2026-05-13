#include "league.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

namespace poker_ppo {

League::League(IPokerEnvironmentFactory& factory,
               const BetConfig& bet_cfg,
               int obs_dim,
               int action_count,
               int hidden_dim,
               int num_layers,
               BetHistoryConfig    hist,
               RoundSummaryConfig  round_summary,
               Config cfg,
               torch::Device device,
               CFVAuxConfig cfv_aux)
    : factory_(factory),
      bet_cfg_(bet_cfg),
      cfg_(cfg),
      obs_dim_(obs_dim),
      action_count_(action_count),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      hist_(hist),
      round_summary_(round_summary),
      cfv_aux_(cfv_aux),
      device_(device) {}

void League::add_anchor(std::unique_ptr<IPolicy> policy) {
    anchors_.push_back(std::move(policy));
}

void League::add_default_anchors() {
    // No-info baseline.
    add_anchor(std::make_unique<UniformPolicy>());

    // Same architecture, orthogonal-init weights — beating this proves
    // training moved off the random-init manifold (vs. just beating noise).
    add_anchor(std::make_unique<NetworkPolicy>(
        make_random_network(), device_, "random_init"));

    add_anchor(std::make_unique<AlwaysCallPolicy>());
    add_anchor(std::make_unique<AlwaysRaisePolicy>());
    add_anchor(std::make_unique<PairAllInPolicy>());
}

ActorCritic League::clone_network(const ActorCritic& src) {
    // Pass cfv_aux_ so the destination has the same CFV-head shape as
    // the trained source — otherwise the parameter-count check below
    // trips when the head is enabled.
    ActorCritic dst(obs_dim_, action_count_, hidden_dim_, num_layers_,
                    hist_, round_summary_, cfv_aux_);
    dst->to(device_);

    torch::NoGradGuard ng;
    auto src_params = src->parameters();
    auto dst_params = dst->parameters();
    TORCH_CHECK(src_params.size() == dst_params.size(),
                "League::clone_network: parameter count mismatch");
    for (size_t i = 0; i < src_params.size(); ++i) {
        dst_params[i].copy_(src_params[i].detach().to(device_));
    }
    auto src_bufs = src->buffers();
    auto dst_bufs = dst->buffers();
    TORCH_CHECK(src_bufs.size() == dst_bufs.size(),
                "League::clone_network: buffer count mismatch");
    for (size_t i = 0; i < src_bufs.size(); ++i) {
        dst_bufs[i].copy_(src_bufs[i].detach().to(device_));
    }
    dst->eval();
    return dst;
}

ActorCritic League::make_random_network() {
    // Same shape as the trained net (including any CFV head) so the
    // random_init anchor stays comparable. The CFV head's outputs aren't
    // used by the anchor's get_action path; it's just there for shape.
    ActorCritic net(obs_dim_, action_count_, hidden_dim_, num_layers_,
                    hist_, round_summary_, cfv_aux_);
    net->to(device_);
    net->eval();
    return net;
}

League::MatchResult League::play_match(IPolicy& a, IPolicy& b) {
    const int P      = cfg_.num_parallel_envs;
    const int target = cfg_.num_hands_per_match;

    MatchResult mr;
    mr.anchor_name      = b.name();
    mr.action_counts_a.assign(action_count_, 0);
    if (P <= 0 || target <= 0) return mr;

    std::vector<std::unique_ptr<IPokerEnvironment>> envs;
    envs.reserve(P);
    for (int i = 0; i < P; ++i) envs.push_back(factory_.create(bet_cfg_));

    // Alternate seats and flip after each hand so HU NLHE position bias cancels.
    std::vector<int> a_seat(P);
    for (int i = 0; i < P; ++i) a_seat[i] = i % 2;

    std::vector<torch::Tensor> obs(P), masks(P);
    for (int i = 0; i < P; ++i) {
        auto r   = envs[i]->reset();
        obs[i]   = r.observation;
        masks[i] = r.legal_action_mask;
    }

    double total_reward_a = 0.0;
    int    wins_a = 0, losses_a = 0, ties_a = 0;
    int    hands_done = 0;

    // IPolicy takes CPU input.
    auto obs_buf  = torch::zeros({P, obs_dim_},      torch::kFloat32);
    auto mask_buf = torch::zeros({P, action_count_}, torch::kFloat32);

    while (hands_done < target) {
        std::vector<int> a_idxs, b_idxs;
        a_idxs.reserve(P); b_idxs.reserve(P);
        for (int i = 0; i < P; ++i) {
            const int acting = envs[i]->current_player();
            (acting == a_seat[i] ? a_idxs : b_idxs).push_back(i);
        }

        std::vector<int> actions(P, 0);

        auto run_batch = [&](IPolicy& policy, const std::vector<int>& idxs) {
            if (idxs.empty()) return;
            const int n = static_cast<int>(idxs.size());
            auto bo = obs_buf.narrow(0, 0, n);
            auto bm = mask_buf.narrow(0, 0, n);
            for (int j = 0; j < n; ++j) {
                bo[j] = obs[idxs[j]];
                bm[j] = masks[idxs[j]];
            }
            auto chosen = policy.select_actions(bo, bm);
            auto acc    = chosen.accessor<int64_t, 1>();
            for (int j = 0; j < n; ++j) {
                actions[idxs[j]] = static_cast<int>(acc[j]);
            }
        };

        run_batch(a, a_idxs);
        run_batch(b, b_idxs);

        // Mode-collapse diagnostic.
        for (int idx : a_idxs) {
            const int act = actions[idx];
            if (act >= 0 && act < action_count_) ++mr.action_counts_a[act];
        }

        for (int i = 0; i < P; ++i) {
            if (hands_done >= target) break;

            auto r = envs[i]->step(actions[i]);
            if (r.done) {
                // Env reward is in seat 0's frame; flip when A is seat 1.
                const float raw      = r.reward;
                const float a_reward = (a_seat[i] == 0) ? raw : -raw;
                total_reward_a += a_reward;
                if      (a_reward > 0.0f) ++wins_a;
                else if (a_reward < 0.0f) ++losses_a;
                else                       ++ties_a;
                ++hands_done;

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

    mr.num_hands     = hands_done;
    mr.avg_reward_a  = static_cast<float>(total_reward_a / hands_done);
    mr.bb_per_hand_a = mr.avg_reward_a * cfg_.bb_per_unit_reward;
    mr.win_rate_a    = (static_cast<float>(wins_a) + 0.5f * ties_a) /
                       static_cast<float>(hands_done);
    return mr;
}

std::vector<League::MatchResult>
League::evaluate(const ActorCritic& trained) {
    // Snapshot so further training can't race with play_match's forwards.
    NetworkPolicy trained_pol(clone_network(trained), device_, "trained");

    std::vector<MatchResult> out;
    out.reserve(anchors_.size());
    for (auto& anchor : anchors_) {
        out.push_back(play_match(trained_pol, *anchor));
    }
    return out;
}

void League::print_results(const std::vector<MatchResult>& rs,
                           std::ostream& os) const {
    // Sorted descending by bb/hand.
    std::vector<int> order(rs.size());
    for (size_t i = 0; i < rs.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return rs[x].bb_per_hand_a > rs[y].bb_per_hand_a; });

    os << "──────────────────────────────────────────────────────────────────\n"
       << "  vs anchor          hands     bb/hand     win%     mode-collapse?\n"
       << "──────────────────────────────────────────────────────────────────\n"
       << std::fixed << std::setprecision(3);
    for (int idx : order) {
        const auto& r = rs[idx];

        // Flag if any one action is >90% of the policy's choices.
        int64_t total = 0, top = 0;
        for (auto c : r.action_counts_a) { total += c; if (c > top) top = c; }
        const float top_pct = (total > 0)
            ? 100.0f * static_cast<float>(top) / static_cast<float>(total) : 0.0f;
        const char* flag = (top_pct > 90.0f) ? "  *MODE*" : "";

        os << "  " << std::setw(16) << std::left << r.anchor_name << std::right
           << std::setw(8)  << r.num_hands
           << std::setw(12) << r.bb_per_hand_a
           << std::setw(9)  << std::setprecision(1) << (r.win_rate_a * 100.0f) << "%"
           << std::setprecision(3)
           << "    top=" << std::setprecision(1) << top_pct << "%" << flag
           << std::setprecision(3) << "\n";
    }
    os.unsetf(std::ios::fixed);
    os << "──────────────────────────────────────────────────────────────────\n";
}

void League::print_results(const std::vector<MatchResult>& rs) const {
    print_results(rs, std::cout);
}

}  // namespace poker_ppo
