#pragma once
//
//  metrics_logger.h — append per-update PPO metrics and per-snapshot
//  league-evaluation results to two CSVs inside a per-run directory.
//
//  The plot_live.py sidecar tails these files and re-renders matplotlib
//  panels every couple of seconds, giving live training curves without
//  pulling TensorBoard / wandb into the build.
//
//  league.csv is in *long* format (one row per (snapshot, anchor) pair)
//  so anchors can be added or removed without changing the schema.
//

#include "league.h"
#include "ppo.h"

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace poker_ppo {

class MetricsLogger {
public:
    // Creates `run_dir` (and parents) if missing, then opens the two CSVs
    // and writes their header rows.
    explicit MetricsLogger(const std::string& run_dir);
    ~MetricsLogger();

    // High-frequency: one row per PPO update.
    void log_update(const PPOTrainer::UpdateStats& s);

    // Sparse: one row per (snapshot, anchor) pair.  Long format, so adding a
    // new anchor doesn't require a schema migration.
    void log_league(int update, int global_step,
                    const std::vector<League::MatchResult>& results);

    const std::string& run_dir() const { return run_dir_; }

private:
    std::string   run_dir_;
    std::ofstream metrics_;
    std::ofstream league_;
    // Both CSVs may be written from different threads (e.g. background eval).
    // Serialise so a partial-line interleave can't corrupt the file.
    std::mutex    mu_;
};

// Convenience: returns "runs/YYYYMMDD_HHMMSS" using local time.
std::string make_run_dir();

}  // namespace poker_ppo
