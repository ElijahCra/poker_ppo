#pragma once
//
//  rollout_pool.h — persistent worker pool for the thread-pool rollout
//  strategy. Generation-based (not a per-call queue): parallel_for bumps
//  a generation counter, wakes all workers, and they steal work via an
//  atomic index. Avoiding a per-job allocation keeps overhead near the
//  syscall floor — the rollout loop calls parallel_for hundreds of times.
//

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace poker_ppo {

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

    // Run body(i) for i in [0, n_jobs). Blocks until all jobs complete.
    // Body must be safe to invoke concurrently from `n_` workers; only the
    // index `i` distinguishes calls.
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

            // Atomic-counter steal: each worker grabs the next index until
            // the range is exhausted. Cheap and naturally load-balances if
            // env.step() costs vary across envs.
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

} // namespace poker_ppo
