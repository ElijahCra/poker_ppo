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
    elo_    .open(run_dir_ + "/elo.csv");

    metrics_ << "update,global_step,policy_loss,value_loss,entropy,"
                "approx_kl,clip_fraction,explained_variance,learning_rate,"
                "rollout_ms,update_ms\n";
    metrics_.flush();

    elo_     << "update,global_step,latest_rating,"
                "vs_initial_winrate,vs_initial_bbhand,"
                "vs_uniform_winrate,vs_uniform_bbhand\n";
    elo_.flush();
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

void MetricsLogger::log_elo(int update, int global_step,
                            float latest_rating,
                            float vs_initial_winrate, float vs_initial_bbhand,
                            float vs_uniform_winrate, float vs_uniform_bbhand) {
    std::lock_guard<std::mutex> lk(mu_);
    elo_ << update << ',' << global_step << ',' << latest_rating << ','
         << vs_initial_winrate << ',' << vs_initial_bbhand << ','
         << vs_uniform_winrate << ',' << vs_uniform_bbhand << '\n';
    elo_.flush();
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
