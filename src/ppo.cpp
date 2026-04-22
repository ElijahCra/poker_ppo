#include "ppo.h"
#include "rollout_coro.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace poker_ppo {

// ═════════════════════════════════════════════════════════════════════════════
// VectorizedEnv
// ═════════════════════════════════════════════════════════════════════════════

VectorizedEnv::VectorizedEnv(IPokerEnvironmentFactory& factory,
                             const BetConfig& cfg, int num_envs) {
    envs_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i)
        envs_.push_back(factory.create(cfg));

    int N = num_envs, D = obs_dim(), A = action_count();
    obs_buf_     = torch::zeros({N, D});
    rewards_buf_ = torch::zeros({N});
    dones_buf_   = torch::zeros({N});
    masks_buf_   = torch::zeros({N, A});
    players_buf_ = torch::zeros({N}, torch::kInt32);
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
    rewards_buf_.zero_();
    dones_buf_.zero_();

    for (int i = 0; i < N; ++i) {
        int acting_player = envs_[i]->current_player();
        auto result = envs_[i]->step(actions[i]);

        float r = result.reward;
        if (acting_player == 1) r = -r;

        if (result.done) {
            rewards_buf_[i] = r;
            dones_buf_[i]   = 1.0f;
            auto rr = envs_[i]->reset();
            obs_buf_[i]     = rr.observation;
            masks_buf_[i]   = rr.legal_action_mask;
            players_buf_[i] = envs_[i]->current_player();
        } else {
            obs_buf_[i]     = result.observation;
            rewards_buf_[i] = r;
            masks_buf_[i]   = result.legal_action_mask;
            players_buf_[i] = envs_[i]->current_player();
        }
    }

    return {obs_buf_.clone(), rewards_buf_.clone(),
            dones_buf_.clone(), masks_buf_.clone(), players_buf_.clone()};
}

// ═════════════════════════════════════════════════════════════════════════════
// PPOTrainer
// ═════════════════════════════════════════════════════════════════════════════

PPOTrainer::PPOTrainer(IPokerEnvironmentFactory& env_factory,
                       const BetConfig& bet_cfg,
                       const PPOConfig& ppo_cfg,
                       torch::Device device)
    : cfg_(ppo_cfg), bet_cfg_(bet_cfg), device_(device),
      network_(nullptr)
{
    // Create vectorized environment
    vec_env_ = std::make_unique<VectorizedEnv>(
        env_factory, bet_cfg, cfg_.num_envs);

    int obs_dim      = vec_env_->obs_dim();
    int action_count = vec_env_->action_count();

    // Create network
    network_ = ActorCritic(obs_dim, action_count, cfg_.hidden_dim, cfg_.num_layers);
    network_->to(device_);

    // Optimiser
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(cfg_.learning_rate));

    // Rollout buffer (lives on device — avoids CPU round-trip each rollout)
    buffer_ = std::make_unique<RolloutBuffer>(
        cfg_.num_steps, cfg_.num_envs, obs_dim, action_count, device_);
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::train() {
    // Initial reset — carry state lives on CPU; the coroutine scheduler
    // batches it to device once per inference call.
    int D = vec_env_->obs_dim();
    int A = vec_env_->action_count();
    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    carry_obs_            = torch::zeros({cfg_.num_envs, D}, f_cpu);
    carry_legal_mask_     = torch::zeros({cfg_.num_envs, A}, f_cpu);
    carry_current_player_ = torch::zeros({cfg_.num_envs}, i32_cpu);
    carry_done_           = torch::zeros({cfg_.num_envs}, f_cpu);

    // reset_all() already resets every env; re-read each env's initial
    // obs/mask/player into carry state.
    vec_env_->reset_all();
    for (int i = 0; i < cfg_.num_envs; ++i) {
        carry_obs_[i]        = vec_env_->env(i).observation();
        carry_legal_mask_[i] = vec_env_->env(i).legal_action_mask();
        carry_current_player_.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(vec_env_->env(i).current_player());
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

        using clock = std::chrono::steady_clock;
        using ms    = std::chrono::duration<double, std::milli>;

        auto t0 = clock::now();
        collect_rollout();
        auto t1 = clock::now();
        update();
        auto t2 = clock::now();

        if (update_idx_ % 10 == 0) {
            std::cout << "[update " << update_idx_ << "]"
                      << "  rollout=" << ms(t1 - t0).count() << "ms"
                      << "  update=" << ms(t2 - t1).count() << "ms\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout() {
    network_->eval();
    torch::NoGradGuard no_grad;

    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    // Reach back into VectorizedEnv for the raw env vector. We need per-env
    // access so the coroutines can step individually.
    std::vector<std::unique_ptr<IPokerEnvironment>> dummy_envs_unused;  // for signature
    // (RolloutScheduler keeps a reference, so we need a stable vector. We'll
    // vend one from VectorizedEnv via a small helper.)

    // Acquire the owning vector of envs from VectorizedEnv.
    auto& envs = vec_env_->envs_mut();

    // Batch-inference tuning. min_batch ≤ num_envs so the processor fires
    // even when not every coroutine has submitted yet; max_wait caps latency
    // for the tail end of a step.
    // Half-wave pipelining: firing on N/2 lets fast-stepping envs resubmit
    // their next request while the processor is still running the forward
    // on the slower half, overlapping env work with inference. Firing on N
    // creates a full-wave barrier with zero overlap — worse despite fewer
    // dispatches. If forward_ns > worker_busy_ns (we're forward-bound),
    // consider dropping further (N/4) to pipeline more aggressively.
    const std::size_t min_batch =
        std::max<std::size_t>(1, static_cast<std::size_t>(cfg_.num_envs) / 4);
    const auto max_wait = std::chrono::microseconds(500);

    // Worker pool: each owns a shard of envs and resumes their coroutines
    // in parallel. Overprovisioning past ~num_envs buys nothing (at most
    // one coro per env can be ready at a time on a given shard). Defaults
    // to 4 — tune based on core count and env-step cost.
    const int num_workers = std::min(
        cfg_.num_envs,
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2));

    auto t0 = clock::now();

    RolloutScheduler scheduler(
        network_, device_,
        envs,
        *buffer_,
        cfg_.num_steps,
        vec_env_->obs_dim(),
        vec_env_->action_count(),
        carry_obs_, carry_legal_mask_, carry_current_player_, carry_done_,
        min_batch, max_wait,
        num_workers);

    scheduler.run();

    auto t1 = clock::now();

    scheduler.flush_to_buffer();

    // GAE bootstrap tensors live on CPU in the scheduler; move once to device.
    auto next_value  = scheduler.bootstrap_value().to(device_);
    auto next_done   = scheduler.bootstrap_done().to(device_);
    auto next_player = scheduler.bootstrap_player().to(device_);

    buffer_->compute_returns(next_value, next_done, next_player,
                             cfg_.gamma, cfg_.gae_lambda);

    auto t2 = clock::now();

    global_step_ += cfg_.num_envs * cfg_.num_steps;

    if (update_idx_ % 10 == 0) {
        const double fwd_ms  = scheduler.forward_ns() / 1e6;
        const int    W       = scheduler.num_workers();
        const double wait_ms = (W > 0)
            ? (scheduler.worker_wait_ns() / 1e6) / static_cast<double>(W)
            : 0.0;
        std::cout << "  rollout breakdown:"
                  << " coro_run=" << ms(t1 - t0).count() << "ms"
                  << " flush+gae=" << ms(t2 - t1).count() << "ms"
                  << "  forward=" << fwd_ms << "ms"
                  << "  worker_wait_avg=" << wait_ms << "ms"
                  << " (W=" << W << ")\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::update() {
    network_->train();

    auto batch = buffer_->flatten();
    int B = batch.obs.size(0);

    // Buffer already lives on device — no copy needed
    auto& b_obs     = batch.obs;
    auto& b_actions = batch.actions;
    auto& b_logp    = batch.log_probs;
    auto  b_adv     = batch.advantages;  // copy: normalisation mutates it
    auto& b_ret     = batch.returns;
    auto& b_val     = batch.values;
    auto& b_masks   = batch.legal_masks;

    auto stat_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto total_policy_loss = torch::zeros({}, stat_opts);
    auto total_value_loss  = torch::zeros({}, stat_opts);
    auto total_entropy     = torch::zeros({}, stat_opts);
    auto total_approx_kl   = torch::zeros({}, stat_opts);
    auto total_clip_frac   = torch::zeros({}, stat_opts);
    int  num_updates = 0;

    for (int epoch = 0; epoch < cfg_.update_epochs; ++epoch) {
        // Shuffle
        auto indices = torch::randperm(B, torch::TensorOptions().dtype(torch::kInt64).device(device_));

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
            // Critic forward: get new value estimates
            auto er = network_->evaluate(mb_obs, mb_masks, mb_actions);

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
                    er.value - mb_val,
                    -cfg_.clip_coef, cfg_.clip_coef);
                auto v_loss_unclipped = (er.value - mb_ret).pow(2);
                auto v_loss_clipped   = (v_clipped - mb_ret).pow(2);
                v_loss = 0.5f * torch::max(v_loss_unclipped,
                                           v_loss_clipped).mean();
            } else {
                v_loss = 0.5f * (er.value - mb_ret).pow(2).mean();
            }

            // ── entropy bonus ───────────────────────────────────────
            auto entropy_loss = er.entropy.mean();

            // ── total loss ──────────────────────────────────────────
            auto loss = pg_loss
                      - cfg_.ent_coef * entropy_loss
                      + cfg_.vf_coef  * v_loss;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(network_->parameters(), cfg_.max_grad_norm);
            optimizer_->step();

            // ── stats (accumulate on device; .item() called once after loop) ──
            {
                torch::NoGradGuard ng;
                total_policy_loss += pg_loss.detach();
                total_value_loss  += v_loss.detach();
                total_entropy     += entropy_loss.detach();
                total_approx_kl   += ((ratio.detach() - 1.0f) - logratio.detach()).mean();
                total_clip_frac   += ((ratio.detach() - 1.0f).abs() > cfg_.clip_coef)
                                     .to(torch::kFloat32).mean();
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
        // Single device→host sync for all stats
        log_cb_(UpdateStats{
            update_idx_,
            global_step_,
            total_policy_loss.item<float>() / n,
            total_value_loss.item<float>()  / n,
            total_entropy.item<float>()     / n,
            total_approx_kl.item<float>()   / n,
            total_clip_frac.item<float>()   / n,
            explained_var,
            lr
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::save(const std::string& path) {
    torch::save(network_, path);
}

void PPOTrainer::load(const std::string& path) {
    torch::load(network_, path);
}

} // namespace poker_ppo
