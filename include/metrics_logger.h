#pragma once
//
//  metrics_logger.h — append PPO/league/BR metrics to per-run CSVs.
//  tools/plot_live.py tails these for live training curves.
//

#include "best_response.h"
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

    // Sparse: one row per BR evaluation.
    void log_best_response(const BestResponseEvaluator::Result& r);

    const std::string& run_dir() const { return run_dir_; }

private:
    std::string   run_dir_;
    std::ofstream metrics_;
    std::ofstream league_;
    std::ofstream br_;
    // Both CSVs may be written from different threads (e.g. background eval).
    // Serialise so a partial-line interleave can't corrupt the file.
    std::mutex    mu_;
};

// Convenience: returns "runs/YYYYMMDD_HHMMSS" using local time.
std::string make_run_dir();

}  // namespace poker_ppo
