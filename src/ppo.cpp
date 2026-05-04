#include "ppo.h"

#include "opponent_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>

namespace poker_ppo {

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

PPOTrainer::PPOTrainer(IPokerEnvironmentFactory& env_factory,
                       torch::Device device)
    : device_(device),
      network_(nullptr)
{
    collector_ = std::make_unique<RolloutCollector>(
        env_factory, bet_cfg_, cfg_.num_envs, cfg_.num_steps, device_);

    const int obs_dim      = collector_->obs_dim();
    const int action_count = collector_->action_count();

    network_ = ActorCritic(obs_dim, action_count,
                           cfg_.hidden_dim, cfg_.num_layers,
                           cfg_.hist, cfg_.round_summary);
    network_->to(device_);

    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(cfg_.learning_rate));

    opp_mgr_ = std::make_unique<OpponentManager>(
        cfg_.opp_pool, obs_dim, action_count,
        cfg_.hidden_dim, cfg_.num_layers,
        cfg_.hist, cfg_.round_summary, device_);
    opp_mgr_->reset_assignments(cfg_.num_envs);
}

PPOTrainer::~PPOTrainer() = default;

int PPOTrainer::opponent_pool_size() const {
    return opp_mgr_ ? opp_mgr_->size() : 0;
}

void PPOTrainer::train() {
    collector_->init_carry();
    opp_mgr_->reset_assignments(cfg_.num_envs);

    const int total_updates = cfg_.num_updates();

    for (update_idx_ = 0; update_idx_ < total_updates; ++update_idx_) {
        // Linear LR anneal with min_lr_frac floor — without the floor the
        // last quarter of training does ~no learning.
        if constexpr (cfg_.anneal_lr) {
            const float frac = 1.0f - static_cast<float>(update_idx_) / total_updates;
            constexpr float floor_frac = cfg_.min_lr_frac > 0.0f ? cfg_.min_lr_frac : 0.0f;
            const float lr = cfg_.learning_rate * std::max(frac, floor_frac);
            for (auto& pg : optimizer_->param_groups())
                static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr);
        }

        using clock = std::chrono::steady_clock;
        using ms    = std::chrono::duration<double, std::milli>;

        auto t0 = clock::now();
        at::set_num_threads(1);
        collector_->collect(strategy_, network_, *opp_mgr_,
                            update_idx_, cfg_.gamma, cfg_.gae_lambda);
        auto t1 = clock::now();
        at::set_num_threads(std::thread::hardware_concurrency());
        auto stats = update();
        auto t2 = clock::now();

        stats.rollout_ms = ms(t1 - t0).count();
        stats.update_ms  = ms(t2 - t1).count();

        if (log_cb_) log_cb_(stats);

        if (update_idx_ % 10 == 0) {
            std::cout << "[update " << update_idx_ << "]"
                      << "  rollout=" << stats.rollout_ms << "ms"
                      << "  update=" << stats.update_ms << "ms";
            if (opp_mgr_->enabled()) {
                std::cout << "  pool=" << opp_mgr_->size()
                          << "/" << opp_mgr_->capacity();
            }
            std::cout << "\n";
        }

        opp_mgr_->maybe_snapshot(update_idx_, network_);
    }
}

void PPOTrainer::collect_rollout_serial() {
    collector_->collect(Strategy::Serial, network_, *opp_mgr_,
                        update_idx_, cfg_.gamma, cfg_.gae_lambda);
}

void PPOTrainer::collect_rollout_threadpool() {
    collector_->collect(Strategy::Threadpool, network_, *opp_mgr_,
                        update_idx_, cfg_.gamma, cfg_.gae_lambda);
}

PPOTrainer::BenchmarkResult
PPOTrainer::benchmark_strategies(
    const std::vector<std::pair<std::string, RolloutFn>>& strategies,
    int iters, int warmup, bool verbose) {
    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    collector_->init_carry();
    opp_mgr_->reset_assignments(cfg_.num_envs);

    const int saved_update_idx = update_idx_;
    update_idx_ = 1;  // not %10 — silences the train() log

    auto run_one = [&](const RolloutFn& fn) {
        auto t0 = clock::now();
        fn(*this);
        return ms(clock::now() - t0).count();
    };

    // Interleaved warmup so each path stabilises caches/threads.
    for (int i = 0; i < warmup; ++i) {
        for (const auto& s : strategies) s.second(*this);
    }

    std::vector<std::vector<double>> samples(strategies.size());
    for (auto& v : samples) v.reserve(iters);

    // Interleave per-iter so paths see a similar env state distribution.
    for (int i = 0; i < iters; ++i) {
        for (size_t s = 0; s < strategies.size(); ++s) {
            samples[s].push_back(run_one(strategies[s].second));
        }
    }

    update_idx_ = saved_update_idx;

    auto pack = [](std::vector<double> v) -> BenchmarkResult::Stats {
        std::sort(v.begin(), v.end());
        const double sum    = std::accumulate(v.begin(), v.end(), 0.0);
        const double mean   = sum / static_cast<double>(v.size());
        const double median = v[v.size() / 2];
        const double p95    = v[std::min(v.size() - 1,
                                         static_cast<size_t>(v.size() * 0.95))];
        return {v.front(), median, mean, p95, v.back()};
    };

    BenchmarkResult r;
    r.num_envs  = cfg_.num_envs;
    r.num_steps = cfg_.num_steps;
    r.iters     = iters;
    r.samples   = cfg_.num_envs * cfg_.num_steps;
    r.by_strategy.reserve(strategies.size());
    for (size_t s = 0; s < strategies.size(); ++s) {
        r.by_strategy.emplace_back(strategies[s].first, pack(samples[s]));
    }

    if (verbose) {
        // Slowest median = baseline → all printed multipliers are ≥ 1×.
        const auto* slowest = &r.by_strategy.front().second;
        for (const auto& kv : r.by_strategy)
            if (kv.second.median_ms > slowest->median_ms) slowest = &kv.second;

        std::cout << "\n══════════ rollout benchmark ══════════\n"
                  << "  iters=" << iters << "  warmup=" << warmup
                  << "  num_envs=" << r.num_envs
                  << "  num_steps=" << r.num_steps
                  << "  samples/rollout=" << r.samples << "\n"
                  << "  ─────────────────────────────────────\n"
                  << std::fixed << std::setprecision(2);
        for (const auto& kv : r.by_strategy) {
            const auto& name = kv.first;
            const auto& s    = kv.second;
            std::cout << "  " << std::setw(12) << std::left << name << std::right
                      << "  min=" << s.min_ms    << "ms"
                      << "  med=" << s.median_ms << "ms"
                      << "  mean=" << s.mean_ms  << "ms"
                      << "  p95="  << s.p95_ms   << "ms"
                      << "  max="  << s.max_ms   << "ms"
                      << "  spd="  << (slowest->median_ms / s.median_ms) << "x"
                      << "  us/samp=" << (s.median_ms * 1e3 / r.samples) << "\n";
        }
        std::cout.unsetf(std::ios::fixed);
        std::cout << "═══════════════════════════════════════\n\n";
    }
    return r;
}

PPOTrainer::BenchmarkResult
PPOTrainer::benchmark_rollouts(int iters, int warmup, bool verbose) {
    return benchmark_strategies({
        {"serial",     [](PPOTrainer& t) { t.collect_rollout_serial();     }},
        {"threadpool", [](PPOTrainer& t) { t.collect_rollout_threadpool(); }},
    }, iters, warmup, verbose);
}

PPOTrainer::UpdateStats PPOTrainer::update() {
    network_->train();

    auto& buffer = collector_->buffer();
    auto batch   = buffer.flatten();
    int  B       = batch.obs.size(0);

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

    // Cosine entropy decay. Held constant across this update's
    // minibatches (only changes between updates) so the loss is well
    // defined within an update.
    const float ent_coef_now = [&]() {
        if constexpr (!cfg_.anneal_ent_coef) {
            return cfg_.ent_coef;
        } else {
            const int total = std::max(1, cfg_.num_updates());
            const float progress = std::min(
                1.0f, static_cast<float>(update_idx_) / static_cast<float>(total));
            const float cosine_factor = 0.5f * (1.0f + std::cos(M_PI * progress));
            constexpr float ent_min =
                cfg_.ent_coef_min < cfg_.ent_coef ? cfg_.ent_coef_min : cfg_.ent_coef;
            return ent_min + cosine_factor * (cfg_.ent_coef - ent_min);
        }
    }();

    for (int epoch = 0; epoch < cfg_.update_epochs; ++epoch) {
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

            if constexpr (cfg_.norm_advantages) {
                if (mb_adv.size(0) > 1) {
                    mb_adv = (mb_adv - mb_adv.mean()) /
                             (mb_adv.std() + kAdvantageEps);
                }
            }

            auto er = network_->evaluate(mb_obs, mb_masks, mb_actions);

            // Clipped surrogate.
            auto logratio = er.log_prob - mb_logp;
            auto ratio    = logratio.exp();

            auto pg_loss1 = -mb_adv * ratio;
            auto pg_loss2 = -mb_adv * torch::clamp(
                ratio, 1.0f - cfg_.clip_coef, 1.0f + cfg_.clip_coef);
            auto pg_loss  = torch::max(pg_loss1, pg_loss2).mean();

            // 16 minibatches per update — branch in the binary is worth
            // killing via if constexpr.
            torch::Tensor v_loss;
            if constexpr (cfg_.clip_vloss) {
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

            auto entropy_loss = er.entropy.mean();

            auto loss = pg_loss
                      - ent_coef_now * entropy_loss
                      + cfg_.vf_coef * v_loss;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(network_->parameters(), cfg_.max_grad_norm);
            optimizer_->step();

            // Accumulate on device; one .item() sync after the loop.
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

    float explained_var;
    {
        torch::NoGradGuard ng;
        auto var_y = b_ret.var();
        auto var_e = (b_ret - b_val).var();
        explained_var = (var_y.item<float>() < kAdvantageEps)
            ? -1.0f
            : 1.0f - var_e.item<float>() / (var_y.item<float>() + kAdvantageEps);
    }

    float n  = static_cast<float>(std::max(1, num_updates));
    float lr = cfg_.learning_rate;
    if constexpr (cfg_.anneal_lr) {
        // Mirror train()'s floor logic so the reported lr matches what
        // the optimiser actually uses.
        const float frac       = 1.0f - static_cast<float>(update_idx_) / cfg_.num_updates();
        constexpr float floor_frac = cfg_.min_lr_frac > 0.0f ? cfg_.min_lr_frac : 0.0f;
        lr = cfg_.learning_rate * std::max(frac, floor_frac);
    }
    return UpdateStats{
        update_idx_,
        collector_->global_step(),
        total_policy_loss.item<float>() / n,
        total_value_loss.item<float>()  / n,
        total_entropy.item<float>()     / n,
        total_approx_kl.item<float>()   / n,
        total_clip_frac.item<float>()   / n,
        explained_var,
        lr,
        0.0,  // rollout_ms — set by train()
        0.0,  // update_ms  — set by train()
    };
}

void PPOTrainer::save(const std::string& path) {
    torch::save(network_, path);
}

void PPOTrainer::load(const std::string& path) {
    torch::load(network_, path);
}

} // namespace poker_ppo
