#include "best_response.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace poker_ppo {

BestResponseEvaluator::BestResponseEvaluator(
    IPokerEnvironmentFactory& factory,
    const BetConfig&    bet_cfg,
    int  obs_dim,
    int  action_count,
    int  hidden_dim,
    int  num_layers,
    BetHistoryConfig    hist,
    RoundSummaryConfig  round_summary,
    BestResponseConfig  cfg,
    torch::Device       device)
    : factory_(factory),
      bet_cfg_(bet_cfg),
      cfg_(cfg),
      obs_dim_(obs_dim),
      action_count_(action_count),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      hist_(hist),
      round_summary_(round_summary),
      device_(device),
      rng_(cfg.seed ? cfg.seed : std::random_device{}()) {

    envs_.reserve(cfg_.num_envs);
    for (int i = 0; i < cfg_.num_envs; ++i)
        envs_.push_back(factory_.create(bet_cfg_));

    buffer_ = std::make_unique<RolloutBuffer>(
        cfg_.num_steps, cfg_.num_envs, obs_dim_, action_count_, device_);
}

void BestResponseEvaluator::init_exploiter() {
    exploiter_ = ActorCritic(obs_dim_, action_count_, hidden_dim_, num_layers_,
                             hist_, round_summary_);
    exploiter_->to(device_);
    optimizer_ = std::make_unique<torch::optim::Adam>(
        exploiter_->parameters(),
        torch::optim::AdamOptions(cfg_.learning_rate).eps(1e-5));
}

// evaluate

BestResponseEvaluator::Result
BestResponseEvaluator::evaluate(const ActorCritic& target,
                                int update, int global_step) {
    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;
    const auto t0 = clock::now();

    // Freeze target once so all seeds play the same opponent.
    ActorCritic frozen_target = clone_actor_critic(
        target, obs_dim_, action_count_, hidden_dim_, num_layers_,
        hist_, round_summary_, device_);

    const int  num_seeds  = std::max(1, cfg_.num_exploiter_seeds);
    const bool multi_seed = num_seeds > 1;

    std::vector<float> bb_per_seed(num_seeds, 0.0f);
    int    best_idx       = 0;
    int    best_num_hands = 0;
    double best_total_rew = 0.0;
    int    best_wins      = 0;
    int    best_ties      = 0;
    float  best_bb        = -std::numeric_limits<float>::infinity();

    for (int s = 0; s < num_seeds; ++s) {
        // Multi-seed always re-inits (each seed must be independent for the
        // max-over-seeds bound to be meaningful). Single-seed honours
        // warm_start.
        if (multi_seed || !cfg_.warm_start || !warm_started_) {
            init_exploiter();
            warm_started_ = true;
        }

        EvalStats es = run_one_seed(frozen_target);
        const float avg = es.num_hands > 0
            ? static_cast<float>(es.total_reward / es.num_hands) : 0.0f;
        const float bb  = avg * cfg_.bb_per_unit_reward;
        bb_per_seed[s] = bb;

        if (bb > best_bb) {
            best_bb        = bb;
            best_idx       = s;
            best_num_hands = es.num_hands;
            best_total_rew = es.total_reward;
            best_wins      = es.wins;
            best_ties      = es.ties;
        }
    }

    // ── Aggregate across seeds ─────────────────────────────────────────
    float bb_sum = 0.0f;
    float bb_min = std::numeric_limits<float>::infinity();
    for (float x : bb_per_seed) {
        bb_sum += x;
        bb_min = std::min(bb_min, x);
    }
    const float bb_mean = bb_sum / static_cast<float>(num_seeds);
    float var_sum = 0.0f;
    for (float x : bb_per_seed) var_sum += (x - bb_mean) * (x - bb_mean);
    const float bb_std = std::sqrt(var_sum / static_cast<float>(num_seeds));

    Result r;
    r.update         = update;
    r.global_step    = global_step;
    r.br_updates_run = cfg_.updates_per_eval;
    // Best-seed (= max-bb) stats: this is the tightest measured lower bound.
    r.num_hands      = best_num_hands;
    r.avg_reward_a   = best_num_hands > 0
                       ? static_cast<float>(best_total_rew / best_num_hands) : 0.0f;
    r.bb_per_hand_a  = best_bb;
    r.win_rate_a     = best_num_hands > 0
                       ? (static_cast<float>(best_wins) + 0.5f * best_ties) /
                         static_cast<float>(best_num_hands) : 0.0f;
    r.num_seeds      = num_seeds;
    r.bb_per_hand_mean = bb_mean;
    r.bb_per_hand_min  = bb_min;
    r.bb_per_hand_std  = bb_std;
    r.wall_ms        = ms(clock::now() - t0).count();
    (void)best_idx;  // tracked above but not exposed; kept for future use
    return r;
}

// run_one_seed — train the (already-initialised) exploiter for
// cfg_.updates_per_eval PPO updates against the frozen target, then run the
// eval-match. The caller (evaluate) is responsible for initialising the
// exploiter before each call so different seeds are independent.

BestResponseEvaluator::EvalStats
BestResponseEvaluator::run_one_seed(ActorCritic& frozen_target) {
    const int N = cfg_.num_envs;
    const int D = obs_dim_;
    const int A = action_count_;

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Per-env exploiter seat — alternates per episode for position bias
    // cancellation. HU NLHE is asymmetric (BB vs SB).
    std::vector<int> exploiter_seat(N);
    for (int i = 0; i < N; ++i) exploiter_seat[i] = i % 2;

    // Reset all envs and seed carry state.
    auto carry_obs    = torch::zeros({N, D}, f_cpu);
    auto carry_mask   = torch::zeros({N, A}, f_cpu);
    auto carry_player = torch::zeros({N},    i32_cpu);
    for (int i = 0; i < N; ++i) {
        auto rr = envs_[i]->reset();
        carry_obs[i]  = rr.observation;
        carry_mask[i] = rr.legal_action_mask;
        carry_player.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(envs_[i]->current_player());
    }

    double total_reward = 0.0;
    int    total_hands  = 0;
    int    wins = 0, ties = 0;

    for (int update_i = 0; update_i < cfg_.updates_per_eval; ++update_i) {
        // ── ROLLOUT ──────────────────────────────────────────────────────
        exploiter_->eval();
        frozen_target->eval();
        buffer_->clear();

        std::vector<PlayerRolloutState> player_state(N);

        auto cur_obs    = carry_obs.to(device_);
        auto cur_mask   = carry_mask.to(device_);
        auto cur_player = carry_player.to(device_);

        {
            torch::NoGradGuard ng;
            for (int step = 0; step < cfg_.num_steps; ++step) {
                // Forward both networks on the full batch — wasteful (one of
                // the two outputs is unused per env), but simpler than
                // splitting the batch by acting seat. The cost is dominated
                // by env stepping for this small MLP, and BR runs only
                // every `eval_every` main updates anyway.
                auto er = exploiter_->get_action(cur_obs, cur_mask);
                auto tr = frozen_target->get_action(cur_obs, cur_mask);

                auto e_actions = er.action.to(torch::kCPU).contiguous();
                auto e_logp    = er.log_prob.to(torch::kCPU).contiguous();
                auto e_value   = er.value.to(torch::kCPU).contiguous();
                auto t_actions = tr.action.to(torch::kCPU).contiguous();

                auto cur_obs_cpu    = cur_obs.to(torch::kCPU).contiguous();
                auto cur_mask_cpu   = cur_mask.to(torch::kCPU).contiguous();
                auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();

                auto ea = e_actions.accessor<int64_t, 1>();
                auto el = e_logp.accessor<float, 1>();
                auto ev = e_value.accessor<float, 1>();
                auto ta = t_actions.accessor<int64_t, 1>();
                auto cp = cur_player_cpu.accessor<int32_t, 1>();

                auto next_obs    = torch::zeros({N, D}, f_cpu);
                auto next_mask   = torch::zeros({N, A}, f_cpu);
                auto next_player = torch::zeros({N},    i32_cpu);
                auto np          = next_player.accessor<int32_t, 1>();

                for (int i = 0; i < N; ++i) {
                    const int  acting           = cp[i];
                    const bool exploiter_acting = (acting == exploiter_seat[i]);

                    int64_t action;
                    if (exploiter_acting) {
                        action = ea[i];
                        // Record only the exploiter's transitions.
                        player_state[i].record_step(
                            acting, i, *buffer_,
                            cur_obs_cpu[i], cur_mask_cpu[i],
                            ea[i], el[i], ev[i]);
                    } else {
                        action = ta[i];  // target acts; not recorded
                    }

                    auto res = envs_[i]->step(static_cast<int>(action));
                    player_state[i].step_reward(res.reward);

                    if (res.done) {
                        player_state[i].flush_on_terminal(i, *buffer_);

                        // Track per-hand reward from the exploiter's
                        // perspective for the bb/hand metric.
                        const float exp_r =
                            (exploiter_seat[i] == 0) ? res.reward : -res.reward;
                        total_reward += exp_r;
                        if      (exp_r > 0.0f) ++wins;
                        else if (exp_r == 0.0f) ++ties;
                        ++total_hands;

                        auto rr = envs_[i]->reset();
                        next_obs[i]   = rr.observation;
                        next_mask[i]  = rr.legal_action_mask;
                        np[i]         = static_cast<int32_t>(envs_[i]->current_player());
                        // Alternate exploiter seat after each completed hand.
                        exploiter_seat[i] = 1 - exploiter_seat[i];
                    } else {
                        next_obs[i]   = res.observation;
                        next_mask[i]  = res.legal_action_mask;
                        np[i]         = static_cast<int32_t>(envs_[i]->current_player());
                    }
                }

                cur_obs    = next_obs.to(device_);
                cur_mask   = next_mask.to(device_);
                cur_player = next_player.to(device_);
            }

            // Drain pending pendings (truncated tails).
            for (int i = 0; i < N; ++i)
                player_state[i].flush_on_rollout_end(i, *buffer_);
        }

        // Truncated tails are bootstrapped at V_next = 0 (treat as terminal).
        // Rationale: the exploiter's value head is trained only on
        // exploiter-acting states; its V at the carry_obs is OOD when the
        // target is about to act, so a clean V=0 bootstrap avoids injecting
        // garbage signal at the tail. This biases the very last transition
        // per (player, env) track, but that's at most one transition out of
        // many in a num_steps=128 rollout.
        auto bs_values   = torch::zeros({2, N}, f_cpu);
        auto bs_terminal = torch::zeros({2, N}, f_cpu);  // 0 = treat as truncation
        // Set terminal=1 so compute_returns picks the V=0 branch outright.
        bs_terminal.fill_(1.0f);
        buffer_->compute_returns(cfg_.gamma, cfg_.gae_lambda,
                                 bs_values, bs_terminal);

        // Roll the carry tensors forward for the next rollout.
        carry_obs    = cur_obs.to(torch::kCPU);
        carry_mask   = cur_mask.to(torch::kCPU);
        carry_player = cur_player.to(torch::kCPU).contiguous();

        // ── PPO UPDATE on exploiter ──────────────────────────────────────
        exploiter_->train();
        auto batch = buffer_->flatten();
        const int B = static_cast<int>(batch.obs.size(0));
        if (B == 0) continue;

        const int minibatch_size = std::max(1, B / std::max(1, cfg_.num_minibatches));

        auto& b_obs     = batch.obs;
        auto& b_actions = batch.actions;
        auto& b_logp    = batch.log_probs;
        auto  b_adv     = batch.advantages;  // by-value: we mutate via norm
        auto& b_ret     = batch.returns;
        auto& b_val     = batch.values;
        auto& b_masks   = batch.legal_masks;

        for (int epoch = 0; epoch < cfg_.update_epochs; ++epoch) {
            auto indices = torch::randperm(
                B, torch::TensorOptions().dtype(torch::kInt64).device(device_));

            for (int start = 0; start < B; start += minibatch_size) {
                const int end = std::min(start + minibatch_size, B);
                auto mb_idx = indices.slice(0, start, end);

                auto mb_obs     = b_obs.index_select(0, mb_idx);
                auto mb_actions = b_actions.index_select(0, mb_idx);
                auto mb_logp    = b_logp.index_select(0, mb_idx);
                auto mb_adv     = b_adv.index_select(0, mb_idx);
                auto mb_ret     = b_ret.index_select(0, mb_idx);
                auto mb_val     = b_val.index_select(0, mb_idx);
                auto mb_masks   = b_masks.index_select(0, mb_idx);

                if (cfg_.norm_advantages && mb_adv.size(0) > 1)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8f);

                auto er2 = exploiter_->evaluate(mb_obs, mb_masks, mb_actions);

                auto logratio = er2.log_prob - mb_logp;
                auto ratio    = logratio.exp();

                auto pg1 = -mb_adv * ratio;
                auto pg2 = -mb_adv * torch::clamp(
                    ratio, 1.0f - cfg_.clip_coef, 1.0f + cfg_.clip_coef);
                auto pg_loss = torch::max(pg1, pg2).mean();

                torch::Tensor v_loss;
                if (cfg_.clip_vloss) {
                    auto v_clipped = mb_val + torch::clamp(
                        er2.value - mb_val, -cfg_.clip_coef, cfg_.clip_coef);
                    auto v_uncl = (er2.value - mb_ret).pow(2);
                    auto v_cl   = (v_clipped - mb_ret).pow(2);
                    v_loss = 0.5f * torch::max(v_uncl, v_cl).mean();
                } else {
                    v_loss = 0.5f * (er2.value - mb_ret).pow(2).mean();
                }

                auto entropy_loss = er2.entropy.mean();
                auto loss = pg_loss
                          - cfg_.ent_coef * entropy_loss
                          + cfg_.vf_coef  * v_loss;

                optimizer_->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(
                    exploiter_->parameters(), cfg_.max_grad_norm);
                optimizer_->step();
            }
        }
    }

    // ── POST-TRAINING EVAL ──────────────────────────────────────────────
    // Run a deterministic, no-learning match between the trained exploiter
    // and the frozen target — this is the canonical BR measurement.
    // Averaging the exploiter's *training* rewards biases the bound
    // downward because it includes the early-rollout phase when the
    // exploiter is still randomly initialised. The post-training match
    // measures only the trained exploiter's bb/hand, matching the spirit
    // of the paper's "average over the last few network updates" reporting
    // (Timbers et al. 2020, Section 4).
    EvalStats es;
    if (cfg_.eval_hands > 0) {
        es = eval_match(frozen_target);
    } else {
        // Fallback: use training-time stats (looser bound; debug only).
        es.num_hands    = total_hands;
        es.wins         = wins;
        es.ties         = ties;
        es.total_reward = total_reward;
    }
    return es;
}

// eval_match — the trained exploiter vs the frozen target, no learning, no
// recording. Partitions envs by acting seat each step (as League does) so
// each network forwards only on the sub-batch that needs its decision —
// avoids running both networks redundantly on every env.

BestResponseEvaluator::EvalStats
BestResponseEvaluator::eval_match(ActorCritic& target) {
    EvalStats stats{};
    const int N = cfg_.num_envs;
    const int A = action_count_;
    const int D = obs_dim_;
    if (N <= 0 || cfg_.eval_hands <= 0) return stats;

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    exploiter_->eval();
    target->eval();
    torch::NoGradGuard ng;

    // Reset envs and seed per-env state.
    std::vector<int> exploiter_seat(N);
    auto cur_obs    = torch::zeros({N, D}, f_cpu);
    auto cur_mask   = torch::zeros({N, A}, f_cpu);
    auto cur_player = torch::zeros({N},    i32_cpu);
    for (int i = 0; i < N; ++i) {
        auto rr = envs_[i]->reset();
        cur_obs[i]    = rr.observation;
        cur_mask[i]   = rr.legal_action_mask;
        cur_player.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(envs_[i]->current_player());
        exploiter_seat[i] = i % 2;  // initial seat alternation
    }

    // Reusable [N, ·] sub-batch staging tensors.
    auto sub_obs  = torch::zeros({N, D}, f_cpu);
    auto sub_mask = torch::zeros({N, A}, f_cpu);

    auto run_net = [&](ActorCritic& net, const std::vector<int>& idxs,
                       std::vector<int64_t>& out_actions) {
        if (idxs.empty()) return;
        const int n = static_cast<int>(idxs.size());
        auto bo = sub_obs.narrow(0, 0, n);
        auto bm = sub_mask.narrow(0, 0, n);
        for (int j = 0; j < n; ++j) {
            bo[j] = cur_obs[idxs[j]];
            bm[j] = cur_mask[idxs[j]];
        }
        auto bo_d = bo.to(device_);
        auto bm_d = bm.to(device_);
        auto ar = net->get_action(bo_d, bm_d);
        auto act_cpu = ar.action.to(torch::kCPU).contiguous();
        auto a = act_cpu.accessor<int64_t, 1>();
        for (int j = 0; j < n; ++j) out_actions[idxs[j]] = a[j];
    };

    std::vector<int64_t> actions(N, 0);

    while (stats.num_hands < cfg_.eval_hands) {
        auto cp = cur_player.accessor<int32_t, 1>();

        std::vector<int> exp_idxs, tgt_idxs;
        exp_idxs.reserve(N); tgt_idxs.reserve(N);
        for (int i = 0; i < N; ++i)
            (cp[i] == exploiter_seat[i] ? exp_idxs : tgt_idxs).push_back(i);

        run_net(exploiter_, exp_idxs, actions);
        run_net(target,     tgt_idxs, actions);

        for (int i = 0; i < N; ++i) {
            if (stats.num_hands >= cfg_.eval_hands) break;
            auto res = envs_[i]->step(static_cast<int>(actions[i]));
            if (res.done) {
                const float exp_r = (exploiter_seat[i] == 0)
                                  ? res.reward : -res.reward;
                stats.total_reward += exp_r;
                if      (exp_r > 0.0f)  ++stats.wins;
                else if (exp_r == 0.0f) ++stats.ties;
                ++stats.num_hands;

                auto rr = envs_[i]->reset();
                cur_obs[i]  = rr.observation;
                cur_mask[i] = rr.legal_action_mask;
                cur_player.accessor<int32_t, 1>()[i] =
                    static_cast<int32_t>(envs_[i]->current_player());
                exploiter_seat[i] = 1 - exploiter_seat[i];
            } else {
                cur_obs[i]  = res.observation;
                cur_mask[i] = res.legal_action_mask;
                cur_player.accessor<int32_t, 1>()[i] =
                    static_cast<int32_t>(envs_[i]->current_player());
            }
        }
    }
    return stats;
}

}  // namespace poker_ppo
