#include "ppo.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace poker_ppo {

// ═════════════════════════════════════════════════════════════════════════════
// VectorizedEnv
// ═════════════════════════════════════════════════════════════════════════════

VectorizedEnv::VectorizedEnv(IPokerEnvironmentFactory& factory,
                             const BetConfig& cfg, int num_envs) {
    envs_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i)
        envs_.push_back(factory.create(cfg));
}

torch::Tensor VectorizedEnv::reset_all() {
    int N = num_envs();
    int D = obs_dim();
    auto obs = torch::zeros({N, D});
    for (int i = 0; i < N; ++i) {
        auto result = envs_[i]->reset();
        obs[i] = result.observation;
    }
    return obs;
}

VectorizedEnv::BatchStepResult
VectorizedEnv::step(const std::vector<int>& actions) {
    int N = num_envs();
    int D = obs_dim();
    int A = action_count();

    auto obs     = torch::zeros({N, D});
    auto rewards = torch::zeros({N});
    auto dones   = torch::zeros({N});
    auto masks   = torch::zeros({N, A});
    auto players = torch::zeros({N}, torch::kInt32);

    for (int i = 0; i < N; ++i) {
        // Track which player is acting *before* the step
        int acting_player = envs_[i]->current_player();

        auto result = envs_[i]->step(actions[i]);

        // Reward from player 1's perspective; flip for player 2 so the single
        // network always sees "my reward" regardless of seat.
        float r = result.reward;
        if (acting_player == 1) r = -r;

        // Auto-reset on done
        if (result.done) {
            rewards[i] = r;
            dones[i]   = 1.0f;
            auto reset_result  = envs_[i]->reset();
            obs[i]    = reset_result.observation;
            masks[i]  = reset_result.legal_action_mask;
            players[i] = envs_[i]->current_player();
        } else {
            obs[i]     = result.observation;
            rewards[i] = r;
            dones[i]   = 0.0f;
            masks[i]   = result.legal_action_mask;
            players[i] = envs_[i]->current_player();
        }
    }

    return {obs, rewards, dones, masks, players};
}

// ═════════════════════════════════════════════════════════════════════════════
// PPOTrainer
// ═════════════════════════════════════════════════════════════════════════════

PPOTrainer::PPOTrainer(IPokerEnvironmentFactory& env_factory,
                       const BetConfig& bet_cfg,
                       const PPOConfig& ppo_cfg,
                       torch::Device device)
    : cfg_(ppo_cfg), bet_cfg_(bet_cfg), device_(device),
      actor_(nullptr), critic_(nullptr)
{
    // Create vectorized environment
    vec_env_ = std::make_unique<VectorizedEnv>(
        env_factory, bet_cfg, cfg_.num_envs);

    int obs_dim      = vec_env_->obs_dim();
    int action_count = vec_env_->action_count();

    // Create separate actor and critic networks
    actor_  = Actor(obs_dim, action_count, cfg_.hidden_dim, cfg_.num_layers);
    critic_ = Critic(obs_dim, cfg_.hidden_dim, cfg_.num_layers);
    actor_->to(device_);
    critic_->to(device_);

    // Single optimiser over both networks' parameters.
    // (Using separate optimisers with different learning rates is also
    // reasonable — just create two Adam instances instead.)
    std::vector<torch::Tensor> all_params;
    for (auto& p : actor_->parameters())  all_params.push_back(p);
    for (auto& p : critic_->parameters()) all_params.push_back(p);

    optimizer_ = std::make_unique<torch::optim::Adam>(
        all_params, torch::optim::AdamOptions(cfg_.learning_rate));

    // Rollout buffer
    buffer_ = std::make_unique<RolloutBuffer>(
        cfg_.num_steps, cfg_.num_envs, obs_dim, action_count);
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::train() {
    // Initial reset
    next_obs_  = vec_env_->reset_all().to(device_);
    next_done_ = torch::zeros({cfg_.num_envs}, device_);

    // Build initial legal masks and player tracking
    int A = vec_env_->action_count();
    next_legal_mask_     = torch::zeros({cfg_.num_envs, A}, device_);
    next_current_player_ = torch::zeros({cfg_.num_envs},
                                        torch::TensorOptions()
                                            .dtype(torch::kInt32)
                                            .device(device_));

    for (int i = 0; i < cfg_.num_envs; ++i) {
        next_legal_mask_[i]     = vec_env_->env(i).legal_action_mask().to(device_);
        next_current_player_[i] = vec_env_->env(i).current_player();
    }

    int total_updates = cfg_.num_updates();

    for (update_idx_ = 0; update_idx_ < total_updates; ++update_idx_) {
        // Anneal learning rate
        if (cfg_.anneal_lr) {
            float frac = 1.0f - static_cast<float>(update_idx_) / total_updates;
            float lr   = cfg_.learning_rate * frac;
            for (auto& pg : optimizer_->param_groups())
                static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr);
        }

        collect_rollout();
        update();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout() {
    actor_->eval();
    critic_->eval();
    torch::NoGradGuard no_grad;

    for (int step = 0; step < cfg_.num_steps; ++step) {
        // Actor: sample actions
        auto ar = actor_->get_action(next_obs_, next_legal_mask_);

        // Critic: estimate values
        auto value = critic_->forward(next_obs_);

        // Convert actions to CPU ints
        auto actions_cpu = ar.action.to(torch::kCPU);
        std::vector<int> actions(cfg_.num_envs);
        for (int i = 0; i < cfg_.num_envs; ++i)
            actions[i] = actions_cpu[i].item<int64_t>();

        // Step environments
        auto [obs, rewards, dones, masks, players] = vec_env_->step(actions);

        // Store in buffer (including which player was acting)
        buffer_->insert(step,
                        next_obs_.cpu(),
                        ar.action.cpu(),
                        ar.log_prob.cpu(),
                        rewards,
                        next_done_.cpu(),
                        value.cpu(),
                        next_legal_mask_.cpu(),
                        next_current_player_.cpu());

        // Advance state
        next_obs_            = obs.to(device_);
        next_done_           = dones.to(device_);
        next_legal_mask_     = masks.to(device_);
        next_current_player_ = players.to(device_);

        global_step_ += cfg_.num_envs;
    }

    // Bootstrap value from critic
    auto next_value = critic_->forward(next_obs_);

    buffer_->compute_returns(next_value.cpu(),
                             next_done_.cpu(),
                             next_current_player_.cpu(),
                             cfg_.gamma, cfg_.gae_lambda);
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::update() {
    actor_->train();
    critic_->train();

    auto batch = buffer_->flatten();
    int B = batch.obs.size(0);

    // Move to device
    auto b_obs     = batch.obs.to(device_);
    auto b_actions = batch.actions.to(device_);
    auto b_logp    = batch.log_probs.to(device_);
    auto b_adv     = batch.advantages.to(device_);
    auto b_ret     = batch.returns.to(device_);
    auto b_val     = batch.values.to(device_);
    auto b_masks   = batch.legal_masks.to(device_);

    float total_policy_loss = 0, total_value_loss = 0, total_entropy = 0;
    float total_approx_kl = 0, total_clip_frac = 0;
    int   num_updates = 0;

    for (int epoch = 0; epoch < cfg_.update_epochs; ++epoch) {
        // Shuffle
        auto indices = torch::randperm(B, torch::kInt64).to(device_);

        for (int start = 0; start < B; start += cfg_.minibatch_size()) {
            int end = std::min(start + cfg_.minibatch_size(), B);
            auto mb_idx = indices.slice(0, start, end);

            auto mb_obs     = b_obs.index_select(0, mb_idx);
            auto mb_actions = b_actions.index_select(0, mb_idx);
            auto mb_logp    = b_logp.index_select(0, mb_idx);
            auto mb_adv     = b_adv.index_select(0, mb_idx);
            auto mb_ret     = b_ret.index_select(0, mb_idx);
            auto mb_val     = b_val.index_select(0, mb_idx);
            auto mb_masks   = b_masks.index_select(0, mb_idx);

            // Normalise advantages
            if (cfg_.norm_advantages && mb_adv.size(0) > 1) {
                mb_adv = (mb_adv - mb_adv.mean()) /
                         (mb_adv.std() + 1e-8f);
            }

            // Actor forward: get new log_probs and entropy for stored actions
            auto er = actor_->evaluate(mb_obs, mb_masks, mb_actions);

            // Critic forward: get new value estimates
            auto new_values = critic_->forward(mb_obs);

            // ── policy loss (clipped surrogate) ─────────────────────
            auto logratio = er.log_prob - mb_logp;
            auto ratio    = logratio.exp();

            auto pg_loss1 = -mb_adv * ratio;
            auto pg_loss2 = -mb_adv * torch::clamp(
                ratio, 1.0f - cfg_.clip_coef, 1.0f + cfg_.clip_coef);
            auto pg_loss  = torch::max(pg_loss1, pg_loss2).mean();

            // ── value loss ──────────────────────────────────────────
            torch::Tensor v_loss;
            if (cfg_.clip_vloss) {
                auto v_clipped = mb_val + torch::clamp(
                    new_values - mb_val,
                    -cfg_.clip_coef, cfg_.clip_coef);
                auto v_loss_unclipped = (new_values - mb_ret).pow(2);
                auto v_loss_clipped   = (v_clipped - mb_ret).pow(2);
                v_loss = 0.5f * torch::max(v_loss_unclipped,
                                           v_loss_clipped).mean();
            } else {
                v_loss = 0.5f * (new_values - mb_ret).pow(2).mean();
            }

            // ── entropy bonus ───────────────────────────────────────
            auto entropy_loss = er.entropy.mean();

            // ── total loss ──────────────────────────────────────────
            auto loss = pg_loss
                      - cfg_.ent_coef * entropy_loss
                      + cfg_.vf_coef  * v_loss;

            optimizer_->zero_grad();
            loss.backward();
            // Clip gradients for both networks jointly
            std::vector<torch::Tensor> all_params;
            for (auto& p : actor_->parameters())  all_params.push_back(p);
            for (auto& p : critic_->parameters()) all_params.push_back(p);
            torch::nn::utils::clip_grad_norm_(all_params, cfg_.max_grad_norm);
            optimizer_->step();

            // ── stats ───────────────────────────────────────────────
            total_policy_loss += pg_loss.item<float>();
            total_value_loss  += v_loss.item<float>();
            total_entropy     += entropy_loss.item<float>();

            {
                torch::NoGradGuard ng;
                auto approx_kl = ((ratio - 1.0f) - logratio).mean();
                total_approx_kl += approx_kl.item<float>();
                auto clipped = ((ratio - 1.0f).abs() > cfg_.clip_coef)
                               .to(torch::kFloat32).mean();
                total_clip_frac += clipped.item<float>();
            }
            ++num_updates;
        }
    }

    // ── explained variance ──────────────────────────────────────────────
    float explained_var;
    {
        torch::NoGradGuard ng;
        auto var_y = b_ret.var();
        auto var_e = (b_ret - b_val).var();
        explained_var = (var_y.item<float>() < 1e-8f)
            ? -1.0f
            : 1.0f - var_e.item<float>() / (var_y.item<float>() + 1e-8f);
    }

    // ── logging callback ────────────────────────────────────────────────
    if (log_cb_) {
        float n = static_cast<float>(num_updates);
        float lr = cfg_.learning_rate;
        if (cfg_.anneal_lr) {
            float frac = 1.0f - static_cast<float>(update_idx_) /
                         cfg_.num_updates();
            lr = cfg_.learning_rate * frac;
        }
        log_cb_(UpdateStats{
            update_idx_,
            global_step_,
            total_policy_loss / n,
            total_value_loss  / n,
            total_entropy     / n,
            total_approx_kl   / n,
            total_clip_frac   / n,
            explained_var,
            lr
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::save(const std::string& path_prefix) {
    torch::save(actor_,  path_prefix + "_actor.pt");
    torch::save(critic_, path_prefix + "_critic.pt");
}

void PPOTrainer::load(const std::string& path_prefix) {
    torch::load(actor_,  path_prefix + "_actor.pt");
    torch::load(critic_, path_prefix + "_critic.pt");
}

} // namespace poker_ppo
