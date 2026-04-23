#pragma once
//
//  metrics_logger.h — append per-update PPO metrics and per-snapshot Elo
//  results to two CSVs inside a per-run directory.
//
//  The plot_live.py sidecar tails these files and re-renders matplotlib
//  panels every couple of seconds, giving live training curves without
//  pulling TensorBoard / wandb into the build.
//

#include "ppo.h"

#include <fstream>
#include <mutex>
#include <string>

namespace poker_ppo {

class MetricsLogger {
public:
    // Creates `run_dir` (and parents) if missing, then opens the two CSVs
    // and writes their header rows.
    explicit MetricsLogger(const std::string& run_dir);
    ~MetricsLogger();

    // High-frequency: one row per PPO update.
    void log_update(const PPOTrainer::UpdateStats& s);

    // Sparse: one row per Elo evaluation snapshot.
    void log_elo(int update, int global_step,
                 float latest_rating,
                 float vs_initial_winrate, float vs_initial_bbhand,
                 float vs_uniform_winrate, float vs_uniform_bbhand);

    const std::string& run_dir() const { return run_dir_; }

private:
    std::string   run_dir_;
    std::ofstream metrics_;
    std::ofstream elo_;
    // Elo logging fires from the elo_future thread; metric logging fires
    // from the trainer thread. Serialise access so a partial-line interleave
    // can't corrupt the file.
    std::mutex    mu_;
};

// Convenience: returns "runs/YYYYMMDD_HHMMSS" using local time.
std::string make_run_dir();

}  // namespace poker_ppo
