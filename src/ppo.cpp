#include "ppo.h"

#include "best_response.h"
#include "opponent_manager.h"

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>

namespace poker_ppo {

namespace {

// Toggle CUDA autocast for the update's forward+loss. bf16 keeps fp32's
// exponent range, so no GradScaler is needed (unlike fp16). autocast also
// runs softmax/log_softmax/loss reductions in fp32, so the policy
// distribution, entropy and KL stay numerically identical to the fp32
// path — only the encoder/tower matmuls drop to bf16. clear_cache() on
// disable drops the cached bf16 weight casts, which go stale after each
// optimiser step.
inline void set_update_autocast(bool on) {
    at::autocast::set_autocast_enabled(at::kCUDA, on);
    if (on)  at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    if (!on) at::autocast::clear_cache();
}

// R-NaD reward-side regularisation. When active, the magnet KL moves from
// the LOSS into the REWARDS (see RolloutCollector::set_rnad), so the
// loss-side MMD term switches off — running both would double-regularise.
// Default from config (η=0.1, adopted via paired 3-seed BR A/B);
// POKER_PPO_RNAD_ETA overrides, 0 restoring the MMD-loss form.
float rnad_eta() {
    static const float eta = [] {
        const char* e = std::getenv("POKER_PPO_RNAD_ETA");
        return e ? static_cast<float>(std::atof(e))
                 : config::kPPOConfig.rnad_eta;
    }();
    return eta;
}
bool rnad_active() { return rnad_eta() > 0.0f; }

bool rnad_anneal() {
    static const bool on = [] {
        const char* e = std::getenv("POKER_PPO_RNAD_ANNEAL");
        if (!e) return config::kPPOConfig.rnad_anneal;
        const std::string v(e);
        return v != "0" && v != "false";
    }();
    return on;
}
float rnad_eta_final() {
    static const float v = [] {
        const char* e = std::getenv("POKER_PPO_RNAD_ETA_FINAL");
        return e ? static_cast<float>(std::atof(e))
                 : config::kPPOConfig.rnad_eta_final;
    }();
    return v;
}

// Opt-in (POKER_PPO_PROFILE) phase profile for update(), mirroring the
// rollout one: lap() syncs CUDA so async kernels are attributed to the
// phase that launched them, which inflates the total vs. the clean path.
// Printed once at trainer teardown.
struct UpdateProfile {
    double flatten_ms = 0, magnet_ms = 0, select_ms = 0, fwd_ms = 0,
           bwd_ms = 0, optim_ms = 0, stats_ms = 0;
    int    updates = 0;

    static bool enabled() {
        static const bool on = std::getenv("POKER_PPO_PROFILE") != nullptr;
        return on;
    }

    struct Timer {
        bool active;
        std::chrono::steady_clock::time_point last;
        explicit Timer(bool a) : active(a) {
            if (active) last = std::chrono::steady_clock::now();
        }
        void lap(double& acc) {
            if (!active) return;
            torch::cuda::synchronize();
            const auto t = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double, std::milli>(t - last).count();
            last = t;
        }
    };

    void print() const {
        if (updates == 0) return;
        const double n = updates;
        const double tot = flatten_ms + magnet_ms + select_ms + fwd_ms
                         + bwd_ms + optim_ms + stats_ms;
        auto row = [&](const char* name, double ms) {
            std::cout << "  " << std::setw(12) << std::left << name << std::right
                      << std::setw(10) << std::fixed << std::setprecision(2)
                      << (ms / n) << " ms/update"
                      << std::setw(8) << std::setprecision(1)
                      << (tot > 0 ? 100.0 * ms / tot : 0.0) << "%\n";
        };
        std::cout << "\n────────── update phase profile ──────────\n"
                  << "  updates=" << updates << "  total="
                  << std::fixed << std::setprecision(2) << (tot / n)
                  << " ms/update\n  ─────────────────────────────────\n";
        row("flatten",  flatten_ms);
        row("magnet",   magnet_ms);
        row("select",   select_ms);
        row("forward",  fwd_ms);
        row("backward", bwd_ms);
        row("optim",    optim_ms);
        row("stats",    stats_ms);
        std::cout << "  ─────────────────────────────────\n";
        std::cout.unsetf(std::ios::fixed);
    }
};

UpdateProfile g_update_profile;

struct MBLossOut {
    torch::Tensor loss;                  // autograd-tracked scalar
    torch::Tensor pg, v, ent, kl, clip;  // detached stat scalars
};

// One minibatch's full PPO loss. Single source of truth shared verbatim by
// the eager path and the CUDA-graph capture so the two cannot drift. The
// caller manages autocast state around this call.
MBLossOut compute_mb_loss(ActorCritic&         network,
                          float                ent_coef_now,
                          const torch::Tensor& mb_obs,
                          const torch::Tensor& mb_masks,
                          const torch::Tensor& mb_actions,
                          const torch::Tensor& mb_logp,
                          torch::Tensor        mb_adv,
                          const torch::Tensor& mb_ret,
                          const torch::Tensor& mb_val,
                          const torch::Tensor& magnet_logp) {
    constexpr const PPOConfig& cfg = config::kPPOConfig;

    if constexpr (cfg.norm_advantages) {
        if (mb_adv.size(0) > 1) {
            mb_adv = (mb_adv - mb_adv.mean()) /
                     (mb_adv.std() + kAdvantageEps);
        }
    }

    auto er = network->evaluate(mb_obs, mb_masks, mb_actions);

    // Clipped surrogate.
    auto logratio = er.log_prob - mb_logp;
    auto ratio    = logratio.exp();

    auto pg_loss1 = -mb_adv * ratio;
    auto pg_loss2 = -mb_adv * torch::clamp(
        ratio, 1.0f - cfg.clip_coef, 1.0f + cfg.clip_coef);
    auto pg_loss  = torch::max(pg_loss1, pg_loss2).mean();

    torch::Tensor v_loss;
    if constexpr (cfg.clip_vloss) {
        auto v_clipped = mb_val + torch::clamp(
            er.value - mb_val, -cfg.clip_coef, cfg.clip_coef);
        auto v_loss_unclipped = (er.value - mb_ret).pow(2);
        auto v_loss_clipped   = (v_clipped - mb_ret).pow(2);
        v_loss = 0.5f * torch::max(v_loss_unclipped,
                                   v_loss_clipped).mean();
    } else {
        v_loss = 0.5f * (er.value - mb_ret).pow(2).mean();
    }

    // POKER_PPO_NORM_ENTROPY=1: divide each state's entropy by ln|legal(s)|
    // before averaging. A constant coefficient on raw entropy pushes harder
    // toward uniform in wide-action states (max entropy spans ln2..ln14
    // across spots); normalising equalises the pressure. OFF by default —
    // Rudolph et al.'s 0.05–0.2 band was tuned on the raw bonus, and the
    // normalised entropy lives in [0,1], so ent_coef needs re-centering
    // (~2.5× higher) and a BR A/B before adopting. The env read is a
    // static branch, so it's baked consistently into the CUDA graph.
    static const bool norm_entropy =
        std::getenv("POKER_PPO_NORM_ENTROPY") != nullptr;
    // POKER_PPO_ENTROPY_SIZE_WEIGHT < 1 removes the raise-multiplicity
    // entropy subsidy (see decomposed_entropy): the bonus keeps rewarding
    // fold/call/raise mixing but no longer pays ~ln(12) for spreading
    // mass across raise sizes — the audited cause of weak-hand
    // aggression. 1.0 (default) is exactly the plain entropy.
    static const float ent_size_weight = [] {
        const char* e = std::getenv("POKER_PPO_ENTROPY_SIZE_WEIGHT");
        return e ? static_cast<float>(std::atof(e))
                 : config::kPPOConfig.entropy_size_weight;
    }();
    torch::Tensor entropy_loss;
    if (ent_size_weight != 1.0f) {
        entropy_loss =
            decomposed_entropy(er.log_probs_all, ent_size_weight).mean();
    } else if (norm_entropy) {
        // clamp_min(2): single-legal-action states have zero entropy
        // anyway; this just keeps the denominator away from ln(1)=0.
        auto max_ent = torch::log(mb_masks.sum(-1).clamp_min(2.0f));
        entropy_loss = (er.entropy / max_ent).mean();
    } else {
        entropy_loss = er.entropy.mean();
    }

    // MMD regularisation: KL(π_θ || ρ) per state, mean over the minibatch.
    // ρ is the magnet — a slowly-refreshed snapshot of π_θ whose log-probs
    // arrive precomputed in magnet_logp. Pulls π_θ toward an older self,
    // which gives the regularised-PG family its last-iterate Nash
    // convergence guarantee (Sokota et al. 2023). Stripped from the binary
    // when cfg.kl_coef == 0 (vanilla PPO).
    torch::Tensor mmd_kl_loss;
    if constexpr (cfg.kl_coef > 0.0f) {
        if (!rnad_active()) {
            // KL(π || ρ) = Σ_a π(a) (log π(a) − log ρ(a)). exp(log π) for
            // the weight keeps gradients flowing through log_probs_all.
            const auto pi = er.log_probs_all.exp();
            const auto kl = (pi * (er.log_probs_all - magnet_logp)).sum(-1);
            mmd_kl_loss   = kl.mean();
        }
    }

    auto loss = pg_loss
              - ent_coef_now * entropy_loss
              + cfg.vf_coef * v_loss;
    if constexpr (cfg.kl_coef > 0.0f) {
        if (mmd_kl_loss.defined()) {
            loss = loss + cfg.kl_coef * mmd_kl_loss;
        }
    }

    torch::NoGradGuard ng;
    auto rd = ratio.detach();
    auto ld = logratio.detach();
    return {loss,
            pg_loss.detach(),
            v_loss.detach(),
            entropy_loss.detach(),
            ((rd - 1.0f) - ld).mean(),
            ((rd - 1.0f).abs() > cfg.clip_coef).to(torch::kFloat32).mean()};
}

}  // namespace

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
    : factory_(env_factory),
      device_(device),
      network_(nullptr)
{
    num_envs_ = cfg_.num_envs;
    if (const char* e = std::getenv("POKER_PPO_NUM_ENVS")) {
        const int v = std::atoi(e);
        if (v > 0) {
            num_envs_ = v;
            std::cout << "[override] num_envs=" << num_envs_
                      << " (benchmark only)\n";
        }
    }

    collector_ = std::make_unique<RolloutCollector>(
        env_factory, bet_cfg_, num_envs_, cfg_.num_steps, device_);

    const int obs_dim      = collector_->obs_dim();
    const int action_count = collector_->action_count();

    network_ = ActorCritic(obs_dim, action_count,
                           cfg_.hidden_dim, cfg_.num_layers,
                           cfg_.hist, cfg_.round_summary);
    network_->to(device_);

    optimizer_ = std::make_unique<ForeachAdam>(
        network_->parameters(), cfg_.learning_rate);

    opp_mgr_ = std::make_unique<OpponentManager>(
        cfg_.opp_pool, obs_dim, action_count,
        cfg_.hidden_dim, cfg_.num_layers,
        cfg_.hist, cfg_.round_summary, device_);
    opp_mgr_->reset_assignments(num_envs_);

    // MMD magnet — initialise from the random-init network so the very
    // first update's KL term is well-defined. Refreshed every
    // cfg_.magnet_update_every updates inside train(). Only allocated
    // when the regulariser is enabled.
    if constexpr (cfg_.kl_coef > 0.0f) {
        magnet_ = clone_actor_critic(
            network_, obs_dim, action_count,
            cfg_.hidden_dim, cfg_.num_layers,
            cfg_.hist, cfg_.round_summary, device_);
    }

    // R-NaD: hand the magnet to the collector so the rollout can compute
    // logρ(a|s) per step and perturb rewards. Needs the magnet to exist
    // (kl_coef > 0 allocates it; the loss-side KL term is then disabled —
    // see compute_mb_loss).
    if (rnad_active()) {
        if (magnet_.is_empty()) {
            std::cerr << "[ppo] POKER_PPO_RNAD_ETA set but no magnet "
                      << "(cfg.kl_coef == 0) — R-NaD disabled\n";
        } else {
            collector_->set_rnad(&magnet_, rnad_eta());
            std::cout << "[ppo] R-NaD reward regularisation: eta="
                      << rnad_eta();
            if (rnad_anneal())
                std::cout << " → " << rnad_eta_final() << " (annealed)";
            std::cout << " (magnet KL moved from loss into rewards)\n";
        }
    }

    // A1 exploiter-augmented self-play: build a best-response trainer whose
    // exploiter we periodically inject into the opponent pool. Short,
    // warm-started budget (it tracks the live policy across injections);
    // eval_hands=0 so evaluate() only TRAINS (no eval match). The pool
    // (with A1 env overrides) then serves these best-responses to the
    // learner. POKER_PPO_A1_UPDATES / _EVERY_STEPS tune cost.
    if (std::getenv("POKER_PPO_A1") != nullptr) {
        BestResponseConfig a1 = config::kBRConfig;
        a1.enabled            = true;
        a1.eval_hands         = 0;       // train only, no match
        a1.num_exploiter_seeds = 1;
        a1.warm_start         = true;    // continue the same exploiter
        a1.updates_per_eval   = 500;
        if (const char* e = std::getenv("POKER_PPO_A1_UPDATES")) {
            const int v = std::atoi(e);
            if (v > 0) a1.updates_per_eval = v;
        }
        a1_exploiter_ = std::make_unique<BestResponseEvaluator>(
            factory_, bet_cfg_, obs_dim, action_count,
            cfg_.hidden_dim, cfg_.num_layers,
            cfg_.hist, cfg_.round_summary, a1, device_);
        std::cout << "[ppo] A1 exploiter-augmented self-play ON: "
                  << a1.updates_per_eval << " warm exploiter updates/inject\n";
    }
}

PPOTrainer::~PPOTrainer() {
    if (UpdateProfile::enabled()) g_update_profile.print();
}

int PPOTrainer::opponent_pool_size() const {
    return opp_mgr_ ? opp_mgr_->size() : 0;
}

void PPOTrainer::train() {
    collector_->init_carry();
    opp_mgr_->reset_assignments(num_envs_);

    // POKER_PPO_TOTAL_STEPS rescales the run length AND the LR/entropy
    // schedule horizon (so e.g. a 1B-step run anneals over 1B, not the
    // configured 600M). sched_updates_ is the schedule denominator used
    // by both the anneal here and the mirror in update().
    int64_t sched_total_steps = cfg_.total_timesteps;
    if (const char* e = std::getenv("POKER_PPO_TOTAL_STEPS")) {
        const long long v = std::atoll(e);
        if (v > 0) {
            sched_total_steps = v;
            std::cout << "[override] total steps=" << v << "\n";
        }
    }
    sched_updates_ = std::max(1,
        static_cast<int>(sched_total_steps / cfg_.batch_size()));

    // POKER_PPO_MAX_STEPS caps the LOOP below the schedule horizon (a
    // prefix — schedules still target sched_updates_), e.g. for A/B arms
    // that exit cleanly right after their data is collected.
    int total_updates = sched_updates_;
    if (const char* e = std::getenv("POKER_PPO_MAX_STEPS")) {
        const long long cap = std::atoll(e);
        if (cap > 0) {
            const int capped = static_cast<int>(
                (cap + cfg_.batch_size() - 1) / cfg_.batch_size());
            total_updates = std::min(total_updates, capped);
            std::cout << "[override] max steps=" << cap
                      << " -> " << total_updates << " updates\n";
        }
    }

    for (update_idx_ = start_update_; update_idx_ < total_updates; ++update_idx_) {
        // Linear LR anneal with min_lr_frac floor — without the floor the
        // last quarter of training does ~no learning. Annealed against the
        // FULL run length (not a POKER_PPO_MAX_STEPS cap), matching the
        // mirror in update() and keeping capped runs prefix-faithful.
        if constexpr (cfg_.anneal_lr) {
            const float frac = 1.0f
                - static_cast<float>(update_idx_) / sched_updates_;
            constexpr float floor_frac = cfg_.min_lr_frac > 0.0f ? cfg_.min_lr_frac : 0.0f;
            const float lr = cfg_.learning_rate * std::max(frac, floor_frac);
            optimizer_->set_lr(lr);
        }

        // Anneal R-NaD η over the schedule horizon (NashPG-style). Cheap
        // CPU-scalar update; not graph-captured. Held at η_final past the
        // horizon (POKER_PPO_MAX_STEPS arms exit before then anyway).
        if (rnad_active() && rnad_anneal() && !magnet_.is_empty()) {
            const float p = std::min(1.0f,
                static_cast<float>(update_idx_) / sched_updates_);
            collector_->set_rnad_eta(
                rnad_eta() * (1.0f - p) + rnad_eta_final() * p);
        }

        using clock = std::chrono::steady_clock;
        using ms    = std::chrono::duration<double, std::milli>;

        auto t0 = clock::now();
        at::set_num_threads(1);
        collector_->collect(strategy_, network_, *opp_mgr_,
                            cfg_.gamma, cfg_.gae_lambda);
        auto t1 = clock::now();
        at::set_num_threads(std::thread::hardware_concurrency());
        auto stats = update();
        auto t2 = clock::now();

        stats.rollout_ms = ms(t1 - t0).count();
        stats.update_ms  = ms(t2 - t1).count();

        if (log_cb_) log_cb_(stats);

        if (update_idx_ % 10 == 0) {
            std::cout << std::fixed << std::setprecision(1)
                      << "[update " << update_idx_ << "]"
                      << "  rollout=" << stats.rollout_ms << "ms"
                      << "  update=" << stats.update_ms << "ms";
            if (opp_mgr_->enabled()) {
                std::cout << "  pool=" << opp_mgr_->size()
                          << "/" << opp_mgr_->capacity();
            }
            std::cout << "\n" << std::defaultfloat << std::setprecision(6);
        }

        const int64_t global_step = collector_->global_step();

        if (a1_exploiter_) {
            // A1: the pool serves trained best-responses, not learner
            // snapshots — so skip maybe_snapshot and instead train+inject an
            // exploiter on cadence. The exploiter is warm (tracks the live
            // policy); each inject costs ~a1.updates_per_eval exploiter
            // updates, so cadence trades pressure freshness vs compute.
            int64_t a1_every = 6'000'000;
            if (const char* e = std::getenv("POKER_PPO_A1_EVERY_STEPS")) {
                const long long v = std::atoll(e);
                if (v > 0) a1_every = v;
            }
            if (global_step - last_a1_inject_step_ >= a1_every) {
                last_a1_inject_step_ = global_step;
                a1_exploiter_->evaluate(network_, update_idx_, global_step);
                opp_mgr_->inject_opponent(a1_exploiter_->exploiter());
                std::cout << "[a1] injected exploiter @ step " << global_step
                          << "  pool=" << opp_mgr_->size() << "\n";
            }
        } else {
            opp_mgr_->maybe_snapshot(global_step, network_);
        }

        // Refresh the MMD magnet on cadence (env steps — invariant to
        // batch shape). Slow refresh = strong anchoring (drives π_θ
        // toward older self); fast refresh = weak anchoring (closer to
        // vanilla PPO). Sokota 2023's grid landed at K ≈ 100 updates of
        // the original 12,288-step batch — see config.h.
        if constexpr (cfg_.kl_coef > 0.0f) {
            if (cfg_.magnet_refresh_steps > 0 &&
                global_step - last_magnet_refresh_step_
                    >= cfg_.magnet_refresh_steps) {
                last_magnet_refresh_step_ = global_step;
                // In place, not a fresh clone: stable parameter storage
                // means any captured CUDA graph that reads the magnet
                // stays valid across refreshes.
                copy_actor_critic_params(network_, magnet_);
            }
        }

        // Periodic resume checkpoint (env-step cadence; interval-crossing).
        if (ckpt_every_steps_ > 0 && !ckpt_dir_.empty() &&
            global_step - last_ckpt_step_ >= ckpt_every_steps_) {
            last_ckpt_step_ = global_step;
            save_checkpoint(ckpt_dir_);
            std::cout << "[checkpoint] @ update " << update_idx_
                      << " step " << global_step << " → " << ckpt_dir_ << "\n";
        }
    }
}

void PPOTrainer::collect_rollout_serial() {
    collector_->collect(Strategy::Serial, network_, *opp_mgr_,
                        cfg_.gamma, cfg_.gae_lambda);
}

void PPOTrainer::collect_rollout_threadpool() {
    collector_->collect(Strategy::Threadpool, network_, *opp_mgr_,
                        cfg_.gamma, cfg_.gae_lambda);
}

PPOTrainer::BenchmarkResult
PPOTrainer::benchmark_strategies(
    const std::vector<std::pair<std::string, RolloutFn>>& strategies,
    int iters, int warmup, bool verbose) {
    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    collector_->init_carry();
    opp_mgr_->reset_assignments(num_envs_);

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
    r.num_envs  = num_envs_;
    r.num_steps = cfg_.num_steps;
    r.iters     = iters;
    r.samples   = num_envs_ * cfg_.num_steps;
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

    // bf16 autocast for the update's heavy matmuls. On by default on CUDA;
    // POKER_PPO_NO_AMP=1 forces the fp32 path for A/B comparison.
    const bool use_amp =
        device_.is_cuda() && std::getenv("POKER_PPO_NO_AMP") == nullptr;

    UpdateProfile::Timer ut(UpdateProfile::enabled());

    auto& buffer = collector_->buffer();
    auto batch   = buffer.flatten();
    int  B       = batch.obs.size(0);
    ut.lap(g_update_profile.flatten_ms);

    auto& b_obs     = batch.obs;
    auto& b_actions = batch.actions;
    auto& b_logp    = batch.log_probs;
    auto  b_adv     = batch.advantages;  // copy: normalisation mutates it
    auto& b_ret     = batch.returns;
    auto& b_val     = batch.values;
    auto& b_masks   = batch.legal_masks;

    // MMD magnet log-probs for the whole batch, computed once. The magnet
    // is frozen for the duration of an update, so the per-minibatch values
    // are just rows of this table — recomputing the magnet forward in all
    // update_epochs × num_minibatches loop bodies was pure waste. [B, A],
    // fp32 (log_softmax stays fp32 under autocast).
    torch::Tensor magnet_logp_all;
    if constexpr (cfg_.kl_coef > 0.0f) {
        // Under R-NaD the magnet KL lives in the rewards instead; no
        // loss-side table needed.
        if (!rnad_active()) {
            torch::NoGradGuard ng;
            if (use_amp) set_update_autocast(true);
            magnet_logp_all = magnet_->masked_log_probs(b_obs, b_masks);
            if (use_amp) set_update_autocast(false);
        }
    }
    ut.lap(g_update_profile.magnet_ms);

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
            const int total = std::max(1, sched_updates_);
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

            UpdateProfile::Timer mt(UpdateProfile::enabled());

            // Graphed path needs exactly minibatch_size rows (static
            // shapes); the partial tail minibatch — B varies per rollout —
            // takes the eager path below.
            const bool graphed =
                (end - start) == cfg_.minibatch_size() &&
                ensure_update_graph(b_obs, b_actions, b_logp, b_adv, b_ret,
                                    b_val, b_masks, magnet_logp_all, mb_idx,
                                    use_amp, ent_coef_now);

            if (graphed) {
                at::index_select_out(ug_obs_,     b_obs,     0, mb_idx);
                at::index_select_out(ug_actions_, b_actions, 0, mb_idx);
                at::index_select_out(ug_logp_,    b_logp,    0, mb_idx);
                at::index_select_out(ug_adv_,     b_adv,     0, mb_idx);
                at::index_select_out(ug_ret_,     b_ret,     0, mb_idx);
                at::index_select_out(ug_val_,     b_val,     0, mb_idx);
                at::index_select_out(ug_masks_,   b_masks,   0, mb_idx);
                if constexpr (cfg_.kl_coef > 0.0f) {
                    if (magnet_logp_all.defined()) {  // off under R-NaD
                        at::index_select_out(ug_mlogp_, magnet_logp_all,
                                             0, mb_idx);
                    }
                }
                mt.lap(g_update_profile.select_ms);

                // fwd + loss + bwd in one launch; grads land in ug_grads_.
                ugraph_->replay();
                mt.lap(g_update_profile.bwd_ms);

                foreach_clip_grads(ug_grads_, cfg_.max_grad_norm);
                optimizer_->step(ug_grads_);
                mt.lap(g_update_profile.optim_ms);

                {
                    torch::NoGradGuard ng;
                    total_policy_loss += ug_pg_;
                    total_value_loss  += ug_vl_;
                    total_entropy     += ug_ent_;
                    total_approx_kl   += ug_kl_;
                    total_clip_frac   += ug_clip_;
                }
                mt.lap(g_update_profile.stats_ms);
            } else {
                auto mb_obs     = b_obs.index_select(0, mb_idx);
                auto mb_actions = b_actions.index_select(0, mb_idx);
                auto mb_logp    = b_logp.index_select(0, mb_idx);
                auto mb_adv     = b_adv.index_select(0, mb_idx);
                auto mb_ret     = b_ret.index_select(0, mb_idx);
                auto mb_val     = b_val.index_select(0, mb_idx);
                auto mb_masks   = b_masks.index_select(0, mb_idx);
                torch::Tensor mb_mlogp;
                if constexpr (cfg_.kl_coef > 0.0f) {
                    if (magnet_logp_all.defined()) {  // off under R-NaD
                        mb_mlogp = magnet_logp_all.index_select(0, mb_idx);
                    }
                }
                mt.lap(g_update_profile.select_ms);

                if (use_amp) set_update_autocast(true);
                auto out = compute_mb_loss(network_, ent_coef_now,
                                           mb_obs, mb_masks, mb_actions,
                                           mb_logp, mb_adv, mb_ret, mb_val,
                                           mb_mlogp);
                // Backward + step run in fp32 (master weights); disable
                // autocast and drop the stale bf16 weight-cast cache first.
                if (use_amp) set_update_autocast(false);
                mt.lap(g_update_profile.fwd_ms);

                optimizer_->zero_grad();
                out.loss.backward();
                mt.lap(g_update_profile.bwd_ms);
                foreach_clip_grad_norm(optimizer_->params(), cfg_.max_grad_norm);
                optimizer_->step();
                mt.lap(g_update_profile.optim_ms);

                // Accumulate on device; one .item() sync after the loop.
                {
                    torch::NoGradGuard ng;
                    total_policy_loss += out.pg;
                    total_value_loss  += out.v;
                    total_entropy     += out.ent;
                    total_approx_kl   += out.kl;
                    total_clip_frac   += out.clip;
                }
                mt.lap(g_update_profile.stats_ms);
            }
            ++num_updates;
        }
    }
    if (UpdateProfile::enabled()) {
        // Periodic print too — SIGTERM (timeout/ctrl-c) skips destructors.
        if (++g_update_profile.updates % 30 == 0) g_update_profile.print();
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
        const float frac       = 1.0f - static_cast<float>(update_idx_) / sched_updates_;
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

bool PPOTrainer::ensure_update_graph(const torch::Tensor& b_obs,
                                     const torch::Tensor& b_actions,
                                     const torch::Tensor& b_logp,
                                     const torch::Tensor& b_adv,
                                     const torch::Tensor& b_ret,
                                     const torch::Tensor& b_val,
                                     const torch::Tensor& b_masks,
                                     const torch::Tensor& magnet_logp_all,
                                     const torch::Tensor& mb_idx,
                                     bool  use_amp,
                                     float ent_coef_now) {
    if (ugraph_state_ == UGraphState::Ready)  return true;
    if (ugraph_state_ == UGraphState::Failed) return false;

    auto fail = [&](const char* why) {
        ugraph_state_ = UGraphState::Failed;
        if (why) {
            std::cout << "[update] minibatch CUDA graph disabled (" << why
                      << ")\n";
        }
        return false;
    };

    if (!device_.is_cuda()) return fail(nullptr);
    if (std::getenv("POKER_PPO_NO_UPDATE_GRAPH") != nullptr) {
        return fail("POKER_PPO_NO_UPDATE_GRAPH");
    }
    if constexpr (cfg_.anneal_ent_coef) {
        // ent_coef_now changes per update and scalars are baked into
        // captured kernels — the graph would freeze the schedule.
        return fail("anneal_ent_coef");
    }

    try {
        const int64_t M = cfg_.minibatch_size();
        const int64_t D = b_obs.size(1);
        const int64_t A = b_masks.size(1);
        auto f_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        auto i_dev = torch::TensorOptions().dtype(torch::kInt64).device(device_);

        // Static inputs (regular allocator; only what's INSIDE the capture
        // lives in the graph's private pool).
        ug_obs_     = torch::zeros({M, D}, f_dev);
        ug_actions_ = torch::zeros({M},    i_dev);
        ug_logp_    = torch::zeros({M},    f_dev);
        ug_adv_     = torch::zeros({M},    f_dev);
        ug_ret_     = torch::zeros({M},    f_dev);
        ug_val_     = torch::zeros({M},    f_dev);
        ug_masks_   = torch::zeros({M, A}, f_dev);
        if constexpr (cfg_.kl_coef > 0.0f) {
            if (magnet_logp_all.defined()) {  // off under R-NaD
                ug_mlogp_ = torch::zeros({M, magnet_logp_all.size(1)}, f_dev);
            }
        }

        // Live minibatch data for warmup + capture (an all-zero legal mask
        // would be degenerate input for log_softmax).
        at::index_select_out(ug_obs_,     b_obs,     0, mb_idx);
        at::index_select_out(ug_actions_, b_actions, 0, mb_idx);
        at::index_select_out(ug_logp_,    b_logp,    0, mb_idx);
        at::index_select_out(ug_adv_,     b_adv,     0, mb_idx);
        at::index_select_out(ug_ret_,     b_ret,     0, mb_idx);
        at::index_select_out(ug_val_,     b_val,     0, mb_idx);
        at::index_select_out(ug_masks_,   b_masks,   0, mb_idx);
        if constexpr (cfg_.kl_coef > 0.0f) {
            if (magnet_logp_all.defined()) {  // off under R-NaD
                at::index_select_out(ug_mlogp_, magnet_logp_all, 0, mb_idx);
            }
        }

        // Autocast cache must be OFF from warmup through capture: a cached
        // bf16 weight cast from warmup would be baked into the graph as a
        // stale constant, while uncached casts are recorded as kernels
        // that re-read the live fp32 weights on every replay (the
        // make_graphed_callables rule). Note: no clear_cache() inside the
        // capture window — cache-off means there is nothing to clear.
        if (use_amp) at::autocast::set_autocast_cache_enabled(false);
        auto amp_on = [&] {
            if (!use_amp) return;
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        };
        auto amp_off = [&] {
            if (use_amp) at::autocast::set_autocast_enabled(at::kCUDA, false);
        };

        const auto& params = optimizer_->params();
        torch::autograd::variable_list param_list(params.begin(), params.end());

        // Forward under autocast, backward outside it — same shape as the
        // eager path. autograd::grad (not .backward()) so gradients land
        // in fresh graph-pool tensors with stable addresses instead of
        // param.grad().
        auto run_once = [&] {
            amp_on();
            auto out = compute_mb_loss(network_, ent_coef_now,
                                       ug_obs_, ug_masks_, ug_actions_,
                                       ug_logp_, ug_adv_, ug_ret_, ug_val_,
                                       ug_mlogp_);
            amp_off();
            return out;
        };

        auto stream = at::cuda::getStreamFromPool();
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            // Warmup (cuBLAS handles, autograd engine init, workspaces).
            for (int i = 0; i < 3; ++i) {
                auto out = run_once();
                (void)torch::autograd::grad({out.loss}, param_list);
            }
            at::cuda::getCurrentCUDAStream().synchronize();

            ugraph_ = std::make_unique<at::cuda::CUDAGraph>();
            ugraph_->capture_begin();
            auto out   = run_once();
            auto grads = torch::autograd::grad({out.loss}, param_list);
            ugraph_->capture_end();

            ug_pg_   = out.pg;
            ug_vl_   = out.v;
            ug_ent_  = out.ent;
            ug_kl_   = out.kl;
            ug_clip_ = out.clip;
            ug_grads_.assign(grads.begin(), grads.end());
        }
        at::autocast::set_autocast_cache_enabled(true);
        torch::cuda::synchronize();

        ugraph_state_ = UGraphState::Ready;
        std::cout << "[update] CUDA-graphed minibatch fwd+bwd "
                  << "(POKER_PPO_NO_UPDATE_GRAPH=1 disables)\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[update] minibatch CUDA graph capture failed, "
                  << "eager fallback: " << e.what() << "\n";
        // Capture may have died inside the autocast window — restore.
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::set_autocast_cache_enabled(true);
        ugraph_.reset();
        ug_grads_.clear();
        ug_obs_ = ug_actions_ = ug_logp_ = ug_adv_ = ug_ret_ = ug_val_
                = ug_masks_ = ug_mlogp_ = ug_pg_ = ug_vl_ = ug_ent_
                = ug_kl_ = ug_clip_ = torch::Tensor();
        ugraph_state_ = UGraphState::Failed;
        return false;
    }
}

void PPOTrainer::save(const std::string& path) {
    torch::save(network_, path);
}

void PPOTrainer::load(const std::string& path) {
    torch::load(network_, path);
}

void PPOTrainer::save_checkpoint(const std::string& dir) {
    namespace fs = std::filesystem;
    // Write to a sibling tmp dir, then rename over `dir` so a crash mid-
    // write never leaves a half-written checkpoint that resume would load.
    const fs::path final_dir = dir;
    const fs::path tmp_dir   = final_dir.string() + ".tmp";
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    fs::create_directories(tmp_dir, ec);

    torch::save(network_, (tmp_dir / "net.pt").string());
    if (!magnet_.is_empty()) {
        torch::save(magnet_, (tmp_dir / "magnet.pt").string());
    }

    // Optimizer moments + counters in one keyed archive.
    torch::serialize::OutputArchive ar;
    const auto& m = optimizer_->exp_avg();
    const auto& v = optimizer_->exp_avg_sq();
    ar.write("update_idx",  torch::tensor(static_cast<int64_t>(update_idx_)));
    ar.write("global_step", torch::tensor(static_cast<int64_t>(collector_->global_step())));
    ar.write("adam_step",   torch::tensor(static_cast<int64_t>(optimizer_->step_count())));
    ar.write("magnet_refresh_step", torch::tensor(last_magnet_refresh_step_));
    ar.write("n",           torch::tensor(static_cast<int64_t>(m.size())));
    for (size_t i = 0; i < m.size(); ++i) {
        ar.write("m" + std::to_string(i), m[i].to(torch::kCPU));
        ar.write("v" + std::to_string(i), v[i].to(torch::kCPU));
    }
    ar.save_to((tmp_dir / "state.pt").string());

    fs::remove_all(final_dir, ec);
    fs::rename(tmp_dir, final_dir, ec);
    if (ec) {  // cross-device or race: fall back to copy
        fs::remove_all(final_dir, ec);
        fs::copy(tmp_dir, final_dir, fs::copy_options::recursive, ec);
        fs::remove_all(tmp_dir, ec);
    }
}

bool PPOTrainer::load_checkpoint(const std::string& dir) {
    namespace fs = std::filesystem;
    const fs::path d = dir;
    if (!fs::exists(d / "net.pt") || !fs::exists(d / "state.pt")) {
        std::cerr << "[resume] no checkpoint at " << dir << "\n";
        return false;
    }

    torch::load(network_, (d / "net.pt").string(), device_);
    if (!magnet_.is_empty() && fs::exists(d / "magnet.pt")) {
        torch::load(magnet_, (d / "magnet.pt").string(), device_);
    }

    torch::serialize::InputArchive ar;
    ar.load_from((d / "state.pt").string(), device_);
    auto read_i64 = [&](const char* key) {
        torch::Tensor t;
        ar.read(key, t);
        return t.item<int64_t>();
    };
    start_update_ = static_cast<int>(read_i64("update_idx"));
    collector_->set_global_step(read_i64("global_step"));
    last_magnet_refresh_step_ = read_i64("magnet_refresh_step");
    last_ckpt_step_ = collector_->global_step();
    const int64_t adam_step = read_i64("adam_step");
    const int64_t n = read_i64("n");

    std::vector<torch::Tensor> m(n), v(n);
    for (int64_t i = 0; i < n; ++i) {
        ar.read("m" + std::to_string(i), m[i]);
        ar.read("v" + std::to_string(i), v[i]);
    }
    optimizer_->load_state(m, v, adam_step);

    std::cout << "[resume] loaded checkpoint @ update " << start_update_
              << " (step " << collector_->global_step()
              << "); optimizer + magnet restored, opponent pool refills "
              << "from empty\n";
    return true;
}

} // namespace poker_ppo
