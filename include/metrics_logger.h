#pragma once
//
// Append PPO/league/BR metrics to per-run CSVs. tools/plot_live.py tails
// these for live training curves.
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
    // Creates run_dir (and parents) if missing, opens CSVs, writes headers.
    explicit MetricsLogger(const std::string& run_dir);
    ~MetricsLogger();

    // One row per PPO update.
    void log_update(const PPOTrainer::UpdateStats& s);

    // One row per (snapshot, anchor). Long format so adding an anchor
    // doesn't need a schema migration.
    void log_league(int update, int global_step,
                    const std::vector<League::MatchResult>& results);

    void log_best_response(const BestResponseEvaluator::Result& r);

    const std::string& run_dir() const { return run_dir_; }

private:
    std::string   run_dir_;
    std::ofstream metrics_;
    std::ofstream league_;
    std::ofstream br_;
    // CSVs may be written from background eval threads — serialise to
    // avoid partial-line interleaving.
    std::mutex    mu_;
};

// "runs/YYYYMMDD_HHMMSS" using local time.
std::string make_run_dir();

}  // namespace poker_ppo
