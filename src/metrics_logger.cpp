#include "metrics_logger.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace poker_ppo {

MetricsLogger::MetricsLogger(const std::string& run_dir) : run_dir_(run_dir) {
    std::filesystem::create_directories(run_dir_);

    metrics_.open(run_dir_ + "/metrics.csv");
    league_ .open(run_dir_ + "/league.csv");
    br_     .open(run_dir_ + "/br.csv");

    metrics_ << "update,global_step,policy_loss,value_loss,entropy,"
                "approx_kl,clip_fraction,explained_variance,learning_rate,"
                "rollout_ms,update_ms\n";
    metrics_.flush();

    // Long format: one row per (snapshot, anchor) pair.  pivot in plot_live.
    league_ << "update,global_step,anchor,num_hands,bb_per_hand,win_rate\n";
    league_.flush();

    // One row per BR evaluation. bb_per_hand is the max-over-seeds (the
    // tightest measured lower bound); the *_mean/min/std columns are
    // diagnostics over the per-seed distribution.
    br_ << "update,global_step,br_updates_run,num_seeds,num_hands,"
           "bb_per_hand,bb_per_hand_mean,bb_per_hand_min,bb_per_hand_std,"
           "win_rate,wall_ms\n";
    br_.flush();
}

MetricsLogger::~MetricsLogger() = default;

void MetricsLogger::log_update(const PPOTrainer::UpdateStats& s) {
    std::lock_guard<std::mutex> lk(mu_);
    metrics_ << s.update << ',' << s.global_step << ','
             << s.policy_loss << ',' << s.value_loss << ','
             << s.entropy << ',' << s.approx_kl << ','
             << s.clip_fraction << ',' << s.explained_variance << ','
             << s.learning_rate << ','
             << s.rollout_ms << ',' << s.update_ms << '\n';
    metrics_.flush();
}

void MetricsLogger::log_league(int update, int global_step,
                               const std::vector<League::MatchResult>& results) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& r : results) {
        league_ << update << ',' << global_step << ','
                << r.anchor_name << ','
                << r.num_hands << ',' << r.bb_per_hand_a << ','
                << r.win_rate_a << '\n';
    }
    league_.flush();
}

void MetricsLogger::log_best_response(const BestResponseEvaluator::Result& r) {
    std::lock_guard<std::mutex> lk(mu_);
    br_ << r.update << ',' << r.global_step << ','
        << r.br_updates_run << ',' << r.num_seeds << ','
        << r.num_hands << ','
        << r.bb_per_hand_a    << ','
        << r.bb_per_hand_mean << ','
        << r.bb_per_hand_min  << ','
        << r.bb_per_hand_std  << ','
        << r.win_rate_a << ',' << r.wall_ms << '\n';
    br_.flush();
}

std::string make_run_dir() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream os;
    os << "runs/" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return os.str();
}

}  // namespace poker_ppo
