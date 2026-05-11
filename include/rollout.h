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

namespace poker_ppo {

class OpponentManager;

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
              float value,
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
    torch::Tensor values_[2];       // [T, N]
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
        float          done     = 0.0f;
    };

    Pending pending[2];
    float   accumulated[2]      = {0.0f, 0.0f};
    float   next_done_flag[2]   = {0.0f, 0.0f};
    bool    tail_was_terminal[2] = {false, false};

    void record_step(int player, int env_idx, RolloutBuffer& buf,
                     torch::Tensor obs, torch::Tensor mask,
                     int64_t action, float log_prob, float value) {
        Pending& p = pending[player];
        if (p.has) {
            buf.push(player, env_idx,
                     p.obs, p.action, p.log_prob,
                     accumulated[player], p.done, p.value, p.mask);
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
                         accumulated[p], pp.done, pp.value, pp.mask);
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
                         accumulated[p], pp.done, pp.value, pp.mask);
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

    // One rollout: fill buffer, bootstrap, compute returns, update carry,
    // advance global_step_. update_idx is forwarded to opp_mgr for warmup
    // and snapshot cadence.
    void collect(Strategy           strategy,
                 ActorCritic&       network,
                 OpponentManager&   opp_mgr,
                 int                update_idx,
                 float              gamma,
                 float              gae_lambda);

    [[nodiscard]] int                   obs_dim()      const noexcept { return vec_env_->obs_dim(); }
    [[nodiscard]] int                   action_count() const noexcept { return vec_env_->action_count(); }
    [[nodiscard]] int                   global_step()  const noexcept { return global_step_; }
    [[nodiscard]] RolloutBuffer&        buffer()       noexcept       { return *buffer_; }
    [[nodiscard]] const RolloutBuffer&  buffer()       const noexcept { return *buffer_; }

private:
    void ensure_step_pool();

    torch::Device                   device_;
    int                             num_envs_;
    int                             num_steps_;
    int                             global_step_ = 0;

    std::unique_ptr<VectorizedEnv>  vec_env_;
    std::unique_ptr<RolloutBuffer>  buffer_;
    std::unique_ptr<StepThreadPool> step_pool_;

    // CPU tensors; consumed by the next rollout via one batched H2D copy.
    torch::Tensor carry_obs_;             // [num_envs, obs_dim]
    torch::Tensor carry_legal_mask_;      // [num_envs, action_count]
    torch::Tensor carry_current_player_;  // [num_envs] int32
    torch::Tensor carry_done_;            // [num_envs]
};

} // namespace poker_ppo
