#pragma once
//
// Rollout-phase machinery:
//   StepThreadPool      persistent worker pool for the parallel strategy
//   RolloutBuffer       per-(player, env) transition storage + GAE
//   PlayerRolloutState  per-env bookkeeping that drives the buffer
//   RolloutCollector    owns VectorizedEnv + buffer + thread pool, runs the loop
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "types.h"

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace at::cuda {
struct CUDAGraph;  // avoid pulling CUDA headers into every consumer
}

namespace poker_ppo {

class OpponentManager;

// Per-phase wall-time accumulator for the rollout loop. Filled only when
// POKER_PPO_PROFILE is set in the environment; printed once at collector
// teardown. Times are summed across all rollouts for one strategy; divide
// by `rollouts` for per-rollout averages.
struct RolloutProfile {
    double infer_ms        = 0.0;  // policy forward (sync'd)
    double d2h_ms          = 0.0;  // device→host copies of step outputs
    double overrides_ms    = 0.0;  // opponent-pool action overrides
    double alloc_next_ms   = 0.0;  // per-step next_obs/mask/player zero-allocs
    double env_step_ms     = 0.0;  // env step + obs build + buffer push
    double terminal_rng_ms = 0.0;  // per-episode opponent reassignment
    double h2d_ms          = 0.0;  // host→device upload of next state
    double returns_ms      = 0.0;  // bootstrap + GAE + carry update
    double total_ms        = 0.0;  // whole collect() call
    int    rollouts        = 0;
};

// Generation-based pool: parallel_for bumps a counter, workers steal via
// an atomic index. Avoids per-job allocations — the rollout loop calls
// parallel_for hundreds of times per training update.
class StepThreadPool {
public:
    explicit StepThreadPool(int n_workers)
        : n_(std::max(1, n_workers))
    {
        threads_.reserve(n_);
        for (int i = 0; i < n_; ++i) {
            threads_.emplace_back([this] { worker_loop(); });
        }
    }

    ~StepThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : threads_) if (t.joinable()) t.join();
    }

    StepThreadPool(const StepThreadPool&)            = delete;
    StepThreadPool& operator=(const StepThreadPool&) = delete;

    int size() const { return n_; }

    // Run body(i) for i in [0, n_jobs). Blocks until done. body must be
    // safe to call concurrently — only `i` distinguishes calls.
    void parallel_for(int n_jobs, std::function<void(int)> body) {
        if (n_jobs <= 0) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            body_   = std::move(body);
            n_jobs_ = n_jobs;
            next_idx_.store(0, std::memory_order_relaxed);
            ++generation_;
        }
        {
            std::lock_guard<std::mutex> lk(done_mu_);
            active_ = n_;
        }
        cv_.notify_all();

        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [this] { return active_ == 0; });
    }

private:
    void worker_loop() {
        uint64_t my_gen = 0;
        while (true) {
            std::function<void(int)> body;
            int n_jobs = 0;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, my_gen] {
                    return stopping_ || generation_ > my_gen;
                });
                if (stopping_) return;
                my_gen = generation_;
                body   = body_;
                n_jobs = n_jobs_;
            }

            while (true) {
                const int i = next_idx_.fetch_add(1, std::memory_order_relaxed);
                if (i >= n_jobs) break;
                body(i);
            }

            {
                std::lock_guard<std::mutex> lk(done_mu_);
                if (--active_ == 0) done_cv_.notify_one();
            }
        }
    }

    const int                n_;
    std::vector<std::thread> threads_;

    std::mutex                 mu_;
    std::condition_variable    cv_;
    bool                       stopping_   = false;
    uint64_t                   generation_ = 0;

    std::function<void(int)>   body_;
    int                        n_jobs_   = 0;
    std::atomic<int>           next_idx_{0};

    std::mutex                 done_mu_;
    std::condition_variable    done_cv_;
    int                        active_   = 0;  // guarded by done_mu_
};

// Per-player transition storage for alternating self-play. Each (player,
// env) trajectory has its own GAE.
//
// Reward attribution: env emits zero-sum r in player-0's frame; we
// accumulate +r for player 0 and -r for player 1 between their actions.
// At a player's next action, the accumulator becomes the previous
// transition's reward.
//
// Rollout-end truncation: the last transition per (player, env) is
// treated as terminal. ≤ 1 biased tail per player per env per rollout;
// decays out across updates.
class RolloutBuffer {
public:
    RolloutBuffer(int num_steps, int num_envs, int obs_dim, int action_count,
                  torch::Device device = torch::kCPU);

    // Reset counts; reuse storage tensors.
    void clear();

    // Thread-safe as long as one thread per (player, env_idx) — naturally
    // satisfied by one-thread-per-env rollouts.
    void push(int player, int env_idx,
              torch::Tensor obs,      // [obs_dim]      CPU or device_
              int64_t action,
              float log_prob,
              float reward,
              float done,
              float value,            // VRPO: Q(s,a); else V(s)
              float vbar,             // VRPO: V̄(s)=Σ π Q; else == value
              torch::Tensor mask);    // [action_count] CPU or device_

    // GAE per (player, env_idx) trajectory.
    //
    // Per (p, e) tail, bootstrap_terminal[p][e] selects:
    //   1.0 → episode terminal, V_next = 0.
    //   0.0 → rollout-end truncation, V_next = bootstrap_values[p][e].
    //
    // bootstrap_values: V(carry_obs) with zero-sum negation for the
    // non-acting player. bootstrap_terminal: from PlayerRolloutState::
    // tail_was_terminal. Both float32 [2, num_envs] CPU.
    void compute_returns(float gamma, float gae_lambda,
                         const torch::Tensor& bootstrap_values,
                         const torch::Tensor& bootstrap_terminal);

    // All valid transitions concatenated across both players and all envs.
    // Returned tensors live on device_.
    struct FlatBatch {
        torch::Tensor obs;          // [B, obs_dim]
        torch::Tensor actions;      // [B]
        torch::Tensor log_probs;    // [B]
        torch::Tensor advantages;   // [B]
        torch::Tensor returns;      // [B]
        torch::Tensor values;       // [B]
        torch::Tensor legal_masks;  // [B, action_count]
    };
    FlatBatch flatten() const;

    int num_envs()     const { return num_envs_; }
    int num_steps()    const { return num_steps_; }
    int action_count() const { return action_count_; }

private:
    int num_steps_, num_envs_, obs_dim_, action_count_;
    torch::Device device_;

    // Capacity is num_steps along time; actual length per (p, e) is counts_.
    torch::Tensor obs_[2];          // [T, N, obs_dim]
    torch::Tensor actions_[2];      // [T, N]            int64
    torch::Tensor log_probs_[2];    // [T, N]
    torch::Tensor rewards_[2];      // [T, N]
    torch::Tensor dones_[2];        // [T, N]
    torch::Tensor values_[2];       // [T, N]   VRPO: Q(s,a); else V(s)
    torch::Tensor vbar_[2];         // [T, N]   VRPO: V̄(s); else == values_
    torch::Tensor legal_masks_[2];  // [T, N, A]
    torch::Tensor advantages_[2];   // [T, N]
    torch::Tensor returns_[2];      // [T, N]

    std::vector<int32_t> counts_[2];  // [N]
};

// Per-env per-player bookkeeping driving the buffer. The rollout loop
// instantiates one per env and calls record_step / step_reward /
// flush_on_terminal / flush_on_rollout_end.
//
// pending[p]:        most recent transition for p whose reward isn't closed
//                    out yet. Closed at p's next action or on termination.
// accumulated[p]:    env reward in p's frame since p's last action.
// next_done_flag[p]: dones flag for p's next recorded transition. Flips
//                    to 1 after termination so the new episode's first
//                    transition carries the boundary marker.
// tail_was_terminal[p]: whether the most recent push for p was followed
//                       by an episode terminal (vs. another action). Read
//                       at rollout-end to pick the right bootstrap V.
struct PlayerRolloutState {
    struct Pending {
        bool           has = false;
        torch::Tensor  obs;
        torch::Tensor  mask;
        int64_t        action   = 0;
        float          log_prob = 0.0f;
        float          value    = 0.0f;
        float          vbar     = 0.0f;
        float          done     = 0.0f;
    };

    Pending pending[2];
    float   accumulated[2]      = {0.0f, 0.0f};
    float   next_done_flag[2]   = {0.0f, 0.0f};
    bool    tail_was_terminal[2] = {false, false};

    void record_step(int player, int env_idx, RolloutBuffer& buf,
                     torch::Tensor obs, torch::Tensor mask,
                     int64_t action, float log_prob, float value, float vbar) {
        Pending& p = pending[player];
        if (p.has) {
            buf.push(player, env_idx,
                     p.obs, p.action, p.log_prob,
                     accumulated[player], p.done, p.value, p.vbar, p.mask);
            accumulated[player] = 0.0f;
            p.has = false;
            // Followed by another action, not a terminal.
            tail_was_terminal[player] = false;
        }
        p.has      = true;
        p.obs      = std::move(obs);
        p.mask     = std::move(mask);
        p.action   = action;
        p.log_prob = log_prob;
        p.value    = value;
        p.vbar     = vbar;
        p.done     = next_done_flag[player];
        next_done_flag[player] = 0.0f;
    }

    // env_reward is in the canonical zero-sum (player-0) frame.
    void step_reward(float env_reward) {
        accumulated[0] += env_reward;
        accumulated[1] -= env_reward;
    }

    void flush_on_terminal(int env_idx, RolloutBuffer& buf) {
        for (int p = 0; p < 2; ++p) {
            Pending& pp = pending[p];
            if (pp.has) {
                buf.push(p, env_idx,
                         pp.obs, pp.action, pp.log_prob,
                         accumulated[p], pp.done, pp.value, pp.vbar, pp.mask);
                pp.has = false;
                tail_was_terminal[p] = true;
            }
        }
        // Reset accumulators unconditionally:
        // - SB-fold walk: one seat never acted this hand.
        // - Pool rollouts: we skip recording the non-learner — its
        //   accumulator must not leak into the next episode where seat
        //   assignments may have flipped.
        accumulated[0] = 0.0f;
        accumulated[1] = 0.0f;
        next_done_flag[0] = 1.0f;
        next_done_flag[1] = 1.0f;
    }

    // Drain still-pending transitions at rollout end. These are
    // truncations, not terminals — the trainer reads tail_was_terminal[p]
    // to pick the right bootstrap V.
    void flush_on_rollout_end(int env_idx, RolloutBuffer& buf) {
        for (int p = 0; p < 2; ++p) {
            Pending& pp = pending[p];
            if (pp.has) {
                buf.push(p, env_idx,
                         pp.obs, pp.action, pp.log_prob,
                         accumulated[p], pp.done, pp.value, pp.vbar, pp.mask);
                accumulated[p] = 0.0f;
                pp.has = false;
                tail_was_terminal[p] = false;
            }
        }
    }
};

// Owns the rollout phase: VectorizedEnv, RolloutBuffer, lazy StepThreadPool,
// per-rollout carry state, the rollout loop (serial/threadpool, single
// shared body), bootstrap + GAE.
class RolloutCollector {
public:
    enum class Strategy { Serial, Threadpool };

    RolloutCollector(IPokerEnvironmentFactory& factory,
                     const BetConfig&          bet_cfg,
                     int                       num_envs,
                     int                       num_steps,
                     torch::Device             device);

    ~RolloutCollector();

    // Reset envs + carry tensors. Call once before training.
    void init_carry();

    // R-NaD reward-side regularisation (Perolat et al. 2022). When set,
    // the rollout also computes the magnet's log-prob of each taken
    // action (5th packed row; the magnet runs INSIDE the CUDA graph,
    // which stays valid across refreshes because the magnet is refreshed
    // in place), and each π-controlled action perturbs rewards zero-sum:
    // actor −η(logπ−logρ), opponent +η(logπ−logρ). The critic then
    // learns the regularised game's values. Pool-snapshot actions are
    // not transformed. Call before the first collect().
    void set_rnad(ActorCritic* magnet, float eta);
    // Update η between rollouts (annealing). Safe any time: η scales a CPU
    // reward term in the step loop, it is not baked into the CUDA graph
    // (which only computes the magnet log-probs).
    void set_rnad_eta(float eta) noexcept { rnad_eta_ = eta; }

    // One rollout: fill buffer, bootstrap, compute returns, update carry,
    // advance global_step_. global_step_ is forwarded to opp_mgr for its
    // step-denominated warmup/snapshot cadences.
    void collect(Strategy           strategy,
                 ActorCritic&       network,
                 OpponentManager&   opp_mgr,
                 float              gamma,
                 float              gae_lambda);

    [[nodiscard]] int                   obs_dim()      const noexcept { return vec_env_->obs_dim(); }
    [[nodiscard]] int                   action_count() const noexcept { return vec_env_->action_count(); }
    [[nodiscard]] int64_t               global_step()  const noexcept { return global_step_; }
    void set_global_step(int64_t s) noexcept { global_step_ = s; }  // resume
    [[nodiscard]] RolloutBuffer&        buffer()       noexcept       { return *buffer_; }
    [[nodiscard]] const RolloutBuffer&  buffer()       const noexcept { return *buffer_; }

private:
    void ensure_step_pool();

    // Lazily capture `network`'s get_action forward into a CUDA graph with
    // static input/output buffers (g_*). The 128 per-step forwards are tiny
    // (B = num_envs) and dominated by kernel-launch overhead — replaying
    // one pre-built graph collapses ~70 launches into 1. Returns true when
    // the graph is ready for this network; false → eager path (CPU device,
    // POKER_PPO_NO_CUDA_GRAPH set, capture failed, or different network).
    // In-place optimiser updates keep parameter storage stable, so one
    // capture stays valid across training updates.
    bool ensure_cuda_graph(ActorCritic& network);

    torch::Device                   device_;
    int                             num_envs_;
    int                             num_steps_;
    int64_t                         global_step_ = 0;

    std::unique_ptr<VectorizedEnv>  vec_env_;
    std::unique_ptr<RolloutBuffer>  buffer_;
    std::unique_ptr<StepThreadPool> step_pool_;

    // CUDA-graphed inference state. graph_net_ keys the capture to one
    // ActorCriticImpl — a different network falls back to eager.
    enum class GraphState { Unset, Ready, Failed };
    GraphState                            graph_state_ = GraphState::Unset;
    const void*                           graph_net_   = nullptr;
    std::unique_ptr<at::cuda::CUDAGraph>  graph_;
    torch::Tensor g_obs_, g_mask_;  // static inputs
    // Static output: [4, N] fp32 {action, log_prob, value, v_bar} packed
    // in-graph so each step pays a single D2H copy (into packed_pin_).
    // R-NaD adds a 5th row: magnet log-prob of the taken action.
    torch::Tensor g_packed_;
    torch::Tensor packed_pin_;      // [4|5, N] pinned CPU staging

    // R-NaD state (set_rnad). Non-owning; the magnet outlives collects.
    ActorCritic* rnad_magnet_ = nullptr;
    float        rnad_eta_    = 0.0f;

    // Opt-in phase profiling (POKER_PPO_PROFILE). Indexed by Strategy so a
    // serial/threadpool A/B keeps the two breakdowns separate.
    bool           profiling_ = false;
    RolloutProfile prof_[2];   // [Serial, Threadpool]

    // CPU tensors; consumed by the next rollout via one batched H2D copy.
    torch::Tensor carry_obs_;             // [num_envs, obs_dim]
    torch::Tensor carry_legal_mask_;      // [num_envs, action_count]
    torch::Tensor carry_current_player_;  // [num_envs] int32
    torch::Tensor carry_done_;            // [num_envs]
};

} // namespace poker_ppo
