#pragma once
//
//  rollout_coro.h — coroutine-based PPO rollout collection with a parallel
//  worker pool.
//
//  Architecture (after worker-pool refactor):
//
//   ┌─ main thread ─┐                                 ┌─ processor ─┐
//   │ spawns N      │ submit req (with env_idx) ────►│ batch N ≥   │
//   │ coros, waits  │                                 │ min_batch,  │
//   │ on done CV    │                                 │ run forward │
//   └───────────────┘                                 └─────┬───────┘
//                                                           │
//                     ┌─────────── route handle by env_idx ─┘
//                     ▼
//   ┌─ worker 0 ─┐  ┌─ worker 1 ─┐ ... ┌─ worker W-1 ─┐
//   │ envs shard │  │ envs shard │     │  envs shard  │
//   │ resumes h, │  │ resumes h, │     │  resumes h,  │
//   │ env.step() │  │ env.step() │     │  env.step()  │
//   │ resubmits  │  │ resubmits  │     │  resubmits   │
//   └────────────┘  └────────────┘     └──────────────┘
//
//  Key invariants:
//   - each env index is owned by exactly one worker ⇒ no env-level locks,
//     no races on scratch[:, env_idx], carry_*[env_idx], bootstrap_*[env_idx]
//   - the coroutine for env i only ever runs on one thread (its owner)
//   - the processor runs on its own thread and exclusively owns the network
//   - batched inference is preserved: all envs' submissions funnel through
//     one global pending queue regardless of owning worker
//
//  Parallelism unlocked: env stepping (55 ms serial before) now runs on W
//  cores concurrently; infer stays hidden behind it on the processor thread.
//

#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"

#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// InferRequest — a single network-forward request from one coroutine.
//   Carries env_idx so the processor can route the resumption to the worker
//   that owns the env.
// ─────────────────────────────────────────────────────────────────────────────
struct InferRequest {
    torch::Tensor obs;          // [obs_dim]      CPU
    torch::Tensor legal_mask;   // [action_count] CPU
    int           env_idx = 0;

    int64_t       action   = 0;
    float         log_prob = 0.0f;
    float         value    = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
// RolloutPromise / RolloutTask — coroutine plumbing.
// ─────────────────────────────────────────────────────────────────────────────
struct RolloutPromise;

class RolloutTask {
public:
    using promise_type = RolloutPromise;
    using handle_type  = std::coroutine_handle<promise_type>;

    explicit RolloutTask(handle_type h) : h_(h) {}
    ~RolloutTask() { if (h_) h_.destroy(); }

    RolloutTask(RolloutTask&& o) noexcept : h_(std::exchange(o.h_, {})) {}
    RolloutTask& operator=(RolloutTask&& o) noexcept {
        if (this != &o) {
            if (h_) h_.destroy();
            h_ = std::exchange(o.h_, {});
        }
        return *this;
    }
    RolloutTask(const RolloutTask&)            = delete;
    RolloutTask& operator=(const RolloutTask&) = delete;

    bool done() const { return !h_ || h_.done(); }

private:
    handle_type h_;
};

struct RolloutPromise {
    RolloutTask get_return_object() {
        return RolloutTask{handle_type::from_promise(*this)};
    }
    // Eager start: the coroutine runs to its first co_await on whichever
    // thread invoked it (main thread during spawn).
    std::suspend_never  initial_suspend() noexcept { return {}; }
    // Stay suspended at final so the owning RolloutTask cleanly destroys
    // the frame when the scheduler tears down.
    std::suspend_always final_suspend()   noexcept { return {}; }
    void return_void()                               {}
    void unhandled_exception()                       { std::terminate(); }

    using handle_type = std::coroutine_handle<RolloutPromise>;
};

// Forward decls.
class RolloutScheduler;

// ─────────────────────────────────────────────────────────────────────────────
// BatchInferProcessor — dedicated inference thread.
//
// Instead of stashing completed pairs in a queue for the main thread to
// drain, `process()` invokes a user-supplied routing callback for each
// completed handle. The scheduler uses this to push handles straight onto
// the ready queue of the worker that owns the env.
// ─────────────────────────────────────────────────────────────────────────────
class BatchInferProcessor {
public:
    using Pair    = std::pair<std::shared_ptr<InferRequest>, std::coroutine_handle<>>;
    // A completed batch is handed to the router as a list of (handle, env_idx)
    // pairs — the router then partitions by owning worker and issues one
    // push per worker instead of one push per handle. Far fewer mutex acquires
    // at the cost of a tiny temporary vector per batch.
    using RouteBatchFn =
        std::function<void(std::vector<std::pair<std::coroutine_handle<>, int>>)>;

    BatchInferProcessor(ActorCritic net, torch::Device device,
                        std::size_t min_batch,
                        std::chrono::microseconds max_wait,
                        RouteBatchFn route)
        : net_(net), device_(device),
          min_batch_(min_batch), max_wait_(max_wait),
          route_(std::move(route)),
          last_batch_(std::chrono::steady_clock::now()) {}

    ~BatchInferProcessor() { stop(); }

    void start() {
        stop_ = false;
        thread_ = std::thread(&BatchInferProcessor::run, this);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void submit(std::shared_ptr<InferRequest> req, std::coroutine_handle<> h) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            pending_.push_back({std::move(req), h});
        }
        cv_.notify_one();
    }

    // Instrumentation — total nanoseconds spent inside process() (forward +
    // stacking + routing). Read + reset from the main thread per rollout.
    int64_t forward_ns() const { return forward_ns_.load(std::memory_order_relaxed); }
    void    reset_forward_ns() { forward_ns_.store(0, std::memory_order_relaxed); }

private:
    void run() {
        while (true) {
            std::vector<Pair> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::microseconds(200), [this] {
                    return stop_ || pending_.size() >= min_batch_;
                });
                if (stop_ && pending_.empty()) return;

                const bool size_trigger = pending_.size() >= min_batch_;
                const bool wait_trigger =
                    !pending_.empty() &&
                    (std::chrono::steady_clock::now() - last_batch_ > max_wait_);
                if (!size_trigger && !wait_trigger) continue;

                batch = std::move(pending_);
                pending_.clear();
                last_batch_ = std::chrono::steady_clock::now();
            }

            process(batch);
            // process() has routed the whole batch via route_; nothing more.
        }
    }

    void process(std::vector<Pair>& batch) {
        const std::size_t B = batch.size();
        if (B == 0) return;

        auto t0 = std::chrono::steady_clock::now();

        std::vector<torch::Tensor> obs_list(B);
        std::vector<torch::Tensor> mask_list(B);
        for (std::size_t i = 0; i < B; ++i) {
            obs_list[i]  = batch[i].first->obs;
            mask_list[i] = batch[i].first->legal_mask;
        }
        auto obs_batch  = torch::stack(obs_list).to(device_);
        auto mask_batch = torch::stack(mask_list).to(device_);

        torch::Tensor action_cpu, log_prob_cpu, value_cpu;
        {
            torch::NoGradGuard ng;
            auto ar = net_->get_action(obs_batch, mask_batch);
            action_cpu   = ar.action.to(torch::kCPU).contiguous();
            log_prob_cpu = ar.log_prob.to(torch::kCPU).contiguous();
            value_cpu    = ar.value.to(torch::kCPU).contiguous();
        }

        auto a  = action_cpu.accessor<int64_t, 1>();
        auto lp = log_prob_cpu.accessor<float,  1>();
        auto v  = value_cpu.accessor<float,     1>();

        // Build one (handle, env_idx) list; router partitions by owning worker.
        std::vector<std::pair<std::coroutine_handle<>, int>> routed;
        routed.reserve(B);
        for (std::size_t i = 0; i < B; ++i) {
            auto& req = batch[i].first;
            req->action   = a[i];
            req->log_prob = lp[i];
            req->value    = v[i];
            routed.emplace_back(batch[i].second, req->env_idx);
        }
        route_(std::move(routed));

        auto t1 = std::chrono::steady_clock::now();
        forward_ns_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    }

    ActorCritic               net_;
    torch::Device             device_;
    std::size_t               min_batch_;
    std::chrono::microseconds max_wait_;
    RouteBatchFn              route_;
    std::atomic<int64_t>      forward_ns_{0};

    std::thread               thread_;
    bool                      stop_{false};   // guarded by mu_

    std::mutex                mu_;
    std::condition_variable   cv_;
    std::vector<Pair>         pending_;
    std::chrono::steady_clock::time_point last_batch_;
};

// ─────────────────────────────────────────────────────────────────────────────
// InferAwaiter — what a rollout coroutine co_awaits to get its action/value.
// Always suspends; the processor hands the handle to a worker thread for
// resumption once the batch forward completes.
// ─────────────────────────────────────────────────────────────────────────────
struct InferAwaiter {
    RolloutScheduler*             scheduler;
    std::shared_ptr<InferRequest> request;

    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) const;   // defined below
    InferRequest& await_resume() const noexcept { return *request; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Worker — one thread that owns a shard of envs and resumes their coroutines.
//
// The worker loop waits on its own ready queue. When the processor routes a
// handle here (via `push`), the worker drains whatever is available and
// resumes each handle in turn. Each `h.resume()` runs the coroutine body up
// to its next co_await: env.step() + construct next InferRequest + submit
// back to the global processor. Then the worker goes back to waiting.
//
// Because each env is assigned to exactly one worker, coroutine frames and
// their associated env / scratch slot are only ever touched by one thread.
// ─────────────────────────────────────────────────────────────────────────────
class Worker {
public:
    using DoneFn = std::function<void()>;

    Worker() = default;

    // on_done is invoked on this worker's thread AFTER h.resume() has
    // returned, for each handle whose coroutine reached final_suspend.
    // Signalling completion from here (not from inside the coroutine body)
    // guarantees the frame is no longer executing when main thread is
    // notified, so destroying the handle from main is race-free.
    void start(DoneFn on_done) {
        on_done_ = std::move(on_done);
        stop_ = false;
        thread_ = std::thread([this] { loop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void push(std::coroutine_handle<> h) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            ready_.push_back(h);
        }
        cv_.notify_one();
    }

    // Batch variant — one lock acquire for an arbitrary number of handles.
    void push_batch(std::vector<std::coroutine_handle<>>&& batch) {
        if (batch.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (ready_.empty()) {
                ready_ = std::move(batch);
            } else {
                ready_.insert(ready_.end(),
                              std::make_move_iterator(batch.begin()),
                              std::make_move_iterator(batch.end()));
            }
        }
        cv_.notify_one();
    }

    // Instrumentation — total nanoseconds the worker thread was blocked on
    // the ready-queue CV (i.e. starved of work).
    int64_t wait_ns() const { return wait_ns_.load(std::memory_order_relaxed); }
    void    reset_wait_ns() { wait_ns_.store(0, std::memory_order_relaxed); }

private:
    void loop() {
        while (true) {
            std::vector<std::coroutine_handle<>> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                auto w0 = std::chrono::steady_clock::now();
                cv_.wait(lk, [this] { return stop_ || !ready_.empty(); });
                auto w1 = std::chrono::steady_clock::now();
                wait_ns_.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count(),
                    std::memory_order_relaxed);
                if (stop_ && ready_.empty()) return;
                batch.swap(ready_);
            }
            for (auto h : batch) {
                if (!h) continue;
                if (!h.done()) h.resume();
                // By the time h.resume() returns, the coroutine has either
                // suspended on its next co_await (not done) or reached
                // final_suspend (done). In the latter case the frame is
                // fully quiescent and can safely be destroyed from any
                // thread — so we signal completion here, not earlier.
                if (h.done() && on_done_) on_done_();
            }
        }
    }

    std::thread                          thread_;
    std::mutex                           mu_;
    std::condition_variable              cv_;
    std::vector<std::coroutine_handle<>> ready_;
    bool                                 stop_{false};  // guarded by mu_
    DoneFn                               on_done_;
    std::atomic<int64_t>                 wait_ns_{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// RolloutScheduler — owns workers + processor, spawns coroutines, waits
// for completion.
// ─────────────────────────────────────────────────────────────────────────────
class RolloutScheduler {
public:
    RolloutScheduler(ActorCritic net,
                     torch::Device device,
                     std::vector<std::unique_ptr<IPokerEnvironment>>& envs,
                     RolloutBuffer& buffer,
                     int num_steps,
                     int obs_dim,
                     int action_count,
                     torch::Tensor& carry_obs,
                     torch::Tensor& carry_mask,
                     torch::Tensor& carry_player,
                     torch::Tensor& carry_done,
                     std::size_t min_batch,
                     std::chrono::microseconds max_wait,
                     int num_workers,
                     float reward_scale = 1.0f)
        : net_(net), device_(device),
          envs_(envs), buffer_(buffer),
          num_steps_(num_steps), num_envs_(static_cast<int>(envs.size())),
          obs_dim_(obs_dim), action_count_(action_count),
          carry_obs_(carry_obs), carry_mask_(carry_mask),
          carry_player_(carry_player), carry_done_(carry_done),
          reward_scale_(reward_scale),
          num_workers_(std::max(1, num_workers)),
          workers_(std::max(1, num_workers)),
          processor_(net, device, min_batch, max_wait,
                     [this](std::vector<std::pair<std::coroutine_handle<>, int>> routed) {
                         // Partition by owning worker, then one push_batch per
                         // worker — turns 32 lock acquires into at most W.
                         std::vector<std::vector<std::coroutine_handle<>>>
                             per_worker(num_workers_);
                         for (auto& [h, env_idx] : routed) {
                             per_worker[owner_of(env_idx)].push_back(h);
                         }
                         for (int w = 0; w < num_workers_; ++w) {
                             if (!per_worker[w].empty()) {
                                 workers_[w].push_batch(std::move(per_worker[w]));
                             }
                         }
                     })
    {
        auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto i_cpu   = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        obs_scratch_         = torch::zeros({num_steps_, num_envs_, obs_dim_}, f_cpu);
        actions_scratch_     = torch::zeros({num_steps_, num_envs_}, i_cpu);
        log_probs_scratch_   = torch::zeros({num_steps_, num_envs_}, f_cpu);
        rewards_scratch_     = torch::zeros({num_steps_, num_envs_}, f_cpu);
        dones_scratch_       = torch::zeros({num_steps_, num_envs_}, f_cpu);
        values_scratch_      = torch::zeros({num_steps_, num_envs_}, f_cpu);
        legal_masks_scratch_ = torch::zeros({num_steps_, num_envs_, action_count_}, f_cpu);
        players_scratch_     = torch::zeros({num_steps_, num_envs_}, i32_cpu);

        bootstrap_value_  = torch::zeros({num_envs_}, f_cpu);
        bootstrap_done_   = torch::zeros({num_envs_}, f_cpu);
        bootstrap_player_ = torch::zeros({num_envs_}, i32_cpu);

        auto on_done = [this] {
            // Called on a worker thread after h.resume() has returned for a
            // coroutine that reached final_suspend. Safe to wake main here:
            // the frame is fully suspended, not mid-execution.
            if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(done_mu_);
                done_cv_.notify_one();
            }
        };
        for (auto& w : workers_) w.start(on_done);
        processor_.start();
    }

    ~RolloutScheduler() {
        // Processor first (no more routing), then drain workers.
        processor_.stop();
        for (auto& w : workers_) w.stop();
    }

    const torch::Tensor& bootstrap_value()  const { return bootstrap_value_;  }
    const torch::Tensor& bootstrap_done()   const { return bootstrap_done_;   }
    const torch::Tensor& bootstrap_player() const { return bootstrap_player_; }

    // ── Instrumentation ─────────────────────────────────────────────────
    // Total ns the processor spent inside process() (forward + routing).
    int64_t forward_ns()   const { return processor_.forward_ns(); }
    // Sum across all workers of time blocked on the ready-queue CV — high
    // numbers mean workers are starved (processor can't feed them fast
    // enough); low numbers mean workers are CPU-bound on env stepping.
    int64_t worker_wait_ns() const {
        int64_t total = 0;
        for (auto& w : workers_) total += w.wait_ns();
        return total;
    }
    int     num_workers() const { return num_workers_; }

    // Spawn N coroutines on the main thread. Each runs eagerly to its first
    // co_await, which submits a request to the processor. The processor
    // routes each resumed handle to a worker thread. Main thread waits on
    // the done CV until every coroutine has completed.
    void run() {
        // Reset per-rollout instrumentation so each breakdown log reflects
        // only this rollout's work.
        processor_.reset_forward_ns();
        for (auto& w : workers_) w.reset_wait_ns();

        active_.store(num_envs_, std::memory_order_release);

        std::vector<RolloutTask> tasks;
        tasks.reserve(num_envs_);
        for (int i = 0; i < num_envs_; ++i) {
            tasks.push_back(env_coro(i));
            // After env_coro(i) returns, the coroutine is suspended at its
            // first co_await with a request already queued on the processor.
        }

        // Wait for every coroutine to reach co_return.
        {
            std::unique_lock<std::mutex> lk(done_mu_);
            done_cv_.wait(lk, [this] {
                return active_.load(std::memory_order_acquire) == 0;
            });
        }
        // Destroying `tasks` now is safe: all frames are at final_suspend.
    }

    // Bulk-copy scratch tensors into the RolloutBuffer (one insert per step).
    void flush_to_buffer() {
        for (int t = 0; t < num_steps_; ++t) {
            buffer_.insert(t,
                obs_scratch_[t].to(device_),
                actions_scratch_[t].to(device_),
                log_probs_scratch_[t].to(device_),
                rewards_scratch_[t].to(device_),
                dones_scratch_[t].to(device_),
                values_scratch_[t].to(device_),
                legal_masks_scratch_[t].to(device_),
                players_scratch_[t].to(device_));
        }
    }

    // Called by InferAwaiter::await_suspend on whichever thread happens to
    // be running the coroutine (main thread for the first submission, the
    // owning worker thread thereafter).
    void submit_inference(std::shared_ptr<InferRequest> req,
                          std::coroutine_handle<> h) {
        processor_.submit(std::move(req), h);
    }

private:
    int owner_of(int env_idx) const {
        // Contiguous shards: env_idx [0, envs_per_worker) → worker 0, etc.
        // Using a per-worker-sized shard keeps cache locality per-thread.
        const int per = (num_envs_ + num_workers_ - 1) / num_workers_;
        int w = env_idx / per;
        if (w >= num_workers_) w = num_workers_ - 1;
        return w;
    }

    RolloutTask env_coro(int i) {
        // Per-env local state — carried across steps; persisted at the end
        // of the rollout into carry_*[i] for the next rollout to pick up.
        torch::Tensor obs        = carry_obs_[i].clone();
        torch::Tensor legal_mask = carry_mask_[i].clone();
        int32_t       player     = carry_player_[i].item<int32_t>();
        float         done_flag  = carry_done_[i].item<float>();

        auto actions_a  = actions_scratch_.accessor<int64_t, 2>();
        auto logp_a     = log_probs_scratch_.accessor<float, 2>();
        auto rewards_a  = rewards_scratch_.accessor<float, 2>();
        auto dones_a    = dones_scratch_.accessor<float, 2>();
        auto values_a   = values_scratch_.accessor<float, 2>();
        auto players_a  = players_scratch_.accessor<int32_t, 2>();

        for (int t = 0; t < num_steps_; ++t) {
            auto req = std::make_shared<InferRequest>();
            req->obs        = obs;
            req->legal_mask = legal_mask;
            req->env_idx    = i;

            auto& r = co_await InferAwaiter{this, req};

            const int64_t action   = r.action;
            const float   log_prob = r.log_prob;
            const float   value    = r.value;
            const int32_t acting_player = player;

            auto env_res = envs_[i]->step(static_cast<int>(action));
            float reward = env_res.reward;
            if (acting_player == 1) reward = -reward;
            reward *= reward_scale_;

            torch::Tensor next_obs, next_mask;
            int32_t next_player;
            float   next_done_flag;
            if (env_res.done) {
                auto reset_res = envs_[i]->reset();
                next_obs       = reset_res.observation;
                next_mask      = reset_res.legal_action_mask;
                next_player    = static_cast<int32_t>(envs_[i]->current_player());
                next_done_flag = 1.0f;
            } else {
                next_obs       = env_res.observation;
                next_mask      = env_res.legal_action_mask;
                next_player    = static_cast<int32_t>(envs_[i]->current_player());
                next_done_flag = 0.0f;
            }

            // Record PRE-step transition at (t, i). Only this worker writes
            // to this slot, so no synchronisation is required.
            obs_scratch_[t][i]         = obs;
            legal_masks_scratch_[t][i] = legal_mask;
            actions_a[t][i]  = action;
            logp_a[t][i]     = log_prob;
            rewards_a[t][i]  = reward;
            dones_a[t][i]    = done_flag;
            values_a[t][i]   = value;
            players_a[t][i]  = acting_player;

            obs        = next_obs;
            legal_mask = next_mask;
            player     = next_player;
            done_flag  = next_done_flag;
        }

        // Bootstrap V(s_T).
        auto req = std::make_shared<InferRequest>();
        req->obs        = obs;
        req->legal_mask = legal_mask;
        req->env_idx    = i;
        auto& r = co_await InferAwaiter{this, req};

        bootstrap_value_.accessor<float, 1>()[i]    = r.value;
        bootstrap_done_.accessor<float, 1>()[i]     = done_flag;
        bootstrap_player_.accessor<int32_t, 1>()[i] = player;

        carry_obs_[i]  = obs;
        carry_mask_[i] = legal_mask;
        carry_player_.accessor<int32_t, 1>()[i] = player;
        carry_done_.accessor<float,   1>()[i]   = done_flag;

        // No completion signal here — the worker emits it after h.resume()
        // returns, guaranteeing the frame is fully suspended before main
        // thread wakes and destroys the handle.
        co_return;
    }

    ActorCritic                                       net_;
    torch::Device                                     device_;
    std::vector<std::unique_ptr<IPokerEnvironment>>&  envs_;
    RolloutBuffer&                                    buffer_;
    int                                               num_steps_;
    int                                               num_envs_;
    int                                               obs_dim_;
    int                                               action_count_;

    torch::Tensor& carry_obs_;
    torch::Tensor& carry_mask_;
    torch::Tensor& carry_player_;
    torch::Tensor& carry_done_;
    float          reward_scale_;

    torch::Tensor obs_scratch_;
    torch::Tensor actions_scratch_;
    torch::Tensor log_probs_scratch_;
    torch::Tensor rewards_scratch_;
    torch::Tensor dones_scratch_;
    torch::Tensor values_scratch_;
    torch::Tensor legal_masks_scratch_;
    torch::Tensor players_scratch_;

    int                                               num_workers_;
    std::vector<Worker>                               workers_;
    BatchInferProcessor                               processor_;

    torch::Tensor                                     bootstrap_value_;
    torch::Tensor                                     bootstrap_done_;
    torch::Tensor                                     bootstrap_player_;

    // Completion handshake.
    std::atomic<int>          active_{0};
    std::mutex                done_mu_;
    std::condition_variable   done_cv_;
};

// Deferred definition — needs full RolloutScheduler type.
inline void InferAwaiter::await_suspend(std::coroutine_handle<> h) const {
    scheduler->submit_inference(request, h);
}

} // namespace poker_ppo
