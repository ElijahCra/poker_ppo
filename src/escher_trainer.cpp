#include "escher_trainer.h"

#include "config.h"
#include "lbr.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace poker_ppo {

namespace {
constexpr int kRolloutEnvs = 512;   // parallel envs for batched inference
}

// ── Reservoir ───────────────────────────────────────────────────────────────
void EscherTrainer::Reservoir::add(Sample s) {
    ++n_;
    if (data.size() < cap_) { data.push_back(std::move(s)); return; }
    std::uniform_int_distribution<long> d(0, n_ - 1);
    long j = d(rng_);
    if (j < static_cast<long>(cap_)) data[static_cast<size_t>(j)] = std::move(s);
}

// ── construction ────────────────────────────────────────────────────────────
EscherTrainer::EscherTrainer(IPokerEnvironmentFactory& factory, EscherConfig cfg,
                             torch::Device device)
    : factory_(factory), cfg_(cfg), device_(device),
      bet_cfg_(config::kBetConfig), rng_(cfg.seed) {
    env_ = factory_.create(bet_cfg_);
    A_ = bet_cfg_.action_count();
    obs_dim_ = env_->obs_dim();

    const auto& pc = config::kPPOConfig;
    auto make = [&] {
        ActorCritic ac(obs_dim_, A_, pc.hidden_dim, pc.num_layers,
                       pc.hist, pc.round_summary);
        ac->to(device_);
        return ac;
    };
    value_  = make();
    regret_ = make();
    avg_    = make();
    value_opt_  = std::make_unique<torch::optim::Adam>(
        value_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    regret_opt_ = std::make_unique<torch::optim::Adam>(
        regret_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    avg_opt_    = std::make_unique<torch::optim::Adam>(
        avg_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    avg_buf_ = std::make_unique<Reservoir>(cfg_.buf_cap, cfg_.seed + 7);
}

// ── σ = RM⁺ on the regret net's actor logits, masked to legal ───────────────
std::vector<float> EscherTrainer::sigma(ActorCritic& net,
                                        const torch::Tensor& logits_row,
                                        const torch::Tensor& mask_row) {
    // logits_row, mask_row are CPU [A]
    std::vector<float> p(A_, 0.0f);
    const float* lg = logits_row.data_ptr<float>();
    const float* mk = mask_row.data_ptr<float>();
    double s = 0.0;
    for (int a = 0; a < A_; ++a)
        if (mk[a] > 0.5f) { float v = lg[a] > 0.f ? lg[a] : 0.f; p[a] = v; s += v; }
    if (s > 1e-12) { for (int a = 0; a < A_; ++a) p[a] /= static_cast<float>(s); }
    else {
        int n = 0; for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) ++n;
        for (int a = 0; a < A_; ++a) p[a] = (mk[a] > 0.5f) ? 1.0f / n : 0.0f;
    }
    return p;
}

int EscherTrainer::sample(const std::vector<float>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    float r = u(rng), acc = 0.f;
    int last = 0;
    for (int a = 0; a < A_; ++a) { if (probs[a] > 0) { acc += probs[a]; last = a; if (r <= acc) return a; } }
    return last;
}

int EscherTrainer::sample_uniform(const torch::Tensor& mask_row, std::mt19937& rng) {
    const float* mk = mask_row.data_ptr<float>();
    std::vector<int> legal;
    for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) legal.push_back(a);
    std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
    return legal[d(rng)];
}

// ── batched rollout engine ──────────────────────────────────────────────────
// Runs kRolloutEnvs envs until `n_traj` terminations. Each step: batch the
// acting obs, forward regret_ (σ) and value_ (Q), then per-env `act(...)`
// chooses the action and records; `finish(i, z_p0)` fires on terminal.
namespace {
struct EnvSlot {
    std::unique_ptr<IPokerEnvironment> env;
    torch::Tensor obs, mask;   // current decision node (CPU [obs_dim]/[A])
    int player = 0;
    float z = 0.0f;            // accumulated P0-frame reward this trajectory
    bool live = false;
};
}  // namespace

template <class Act, class Finish>
static void run_rollout(IPokerEnvironmentFactory& factory, const BetConfig& bet,
                        int obs_dim, int A, torch::Device device, int n_envs,
                        int n_traj, ActorCritic& regret, ActorCritic& value,
                        Act&& act, Finish&& finish) {
    std::vector<EnvSlot> slots(n_envs);
    for (auto& s : slots) {
        s.env = factory.create(bet);
        auto r = s.env->reset();
        s.obs = r.observation; s.mask = r.legal_action_mask;
        s.player = s.env->current_player(); s.z = 0.f; s.live = true;
    }
    int finished = 0;
    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    while (finished < n_traj) {
        // batch obs of all live slots
        std::vector<int> idx;
        idx.reserve(n_envs);
        for (int i = 0; i < n_envs; ++i) if (slots[i].live) idx.push_back(i);
        if (idx.empty()) break;
        auto obs_b = torch::empty({(long)idx.size(), obs_dim}, opts);
        for (size_t k = 0; k < idx.size(); ++k) obs_b[k] = slots[idx[k]].obs;
        obs_b = obs_b.to(device);
        torch::Tensor lg, q;
        { torch::NoGradGuard ng;
          lg = regret->forward(obs_b).first.to(torch::kCPU);
          q  = value->forward(obs_b).second.to(torch::kCPU); }
        for (size_t k = 0; k < idx.size(); ++k) {
            int i = idx[k];
            auto& s = slots[i];
            int a = act(i, s.player, s.obs, s.mask, lg[k], q[k]);
            auto r = s.env->step(a);
            s.z += r.reward;
            if (r.done) {
                finish(i, s.z);
                ++finished;
                if (finished >= n_traj) { s.live = false; continue; }
                auto rr = s.env->reset();
                s.obs = rr.observation; s.mask = rr.legal_action_mask;
                s.player = s.env->current_player(); s.z = 0.f;
            } else {
                s.obs = r.observation; s.mask = r.legal_action_mask;
                s.player = s.env->current_player();
            }
        }
    }
}

// ── value phase: MC under current σ (both seats), cleared each iter ─────────
void EscherTrainer::collect_and_train_value() {
    value_smp_.clear();
    // per-trajectory pending nodes: (sample_index, acting_player)
    std::vector<std::vector<std::pair<size_t, int>>> pending(kRolloutEnvs);

    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        int a = sample(sig, rng_);
        size_t si = value_smp_.size();
        value_smp_.push_back({obs.clone(), {}, a, 0.0f, 1.0f});
        pending[i].push_back({si, player});
        return a;
    };
    auto finish = [&](int i, float z_p0) {
        for (auto& [si, p] : pending[i])
            value_smp_[si].z = (p == 0) ? z_p0 : -z_p0;  // acting-player frame
        pending[i].clear();
    };
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.value_traj, regret_, value_, act, finish);

    // train Q(obs, a_taken) -> signed terminal utility
    if (value_smp_.empty()) return;
    std::uniform_int_distribution<size_t> pick(0, value_smp_.size() - 1);
    for (int step = 0; step < cfg_.value_steps; ++step) {
        int B = std::min<int>(cfg_.batch_size, (int)value_smp_.size());
        auto X = torch::empty({B, obs_dim_});
        auto act_idx = torch::empty({B}, torch::kLong);
        auto y = torch::empty({B});
        for (int b = 0; b < B; ++b) {
            const auto& s = value_smp_[pick(rng_)];
            X[b] = s.feat; act_idx[b] = s.action; y[b] = s.z;
        }
        X = X.to(device_); act_idx = act_idx.to(device_); y = y.to(device_);
        value_opt_->zero_grad();
        auto qall = value_->forward(X).second;                       // [B, A]
        auto pred = qall.gather(1, act_idx.unsqueeze(1)).squeeze(1);  // Q(s,a)
        auto loss = torch::mse_loss(pred, y);
        loss.backward();
        value_opt_->step();
    }
}

// ── regret phase: fixed sampler (traverser ~ uniform, opp ~ σ), value-fn Q ──
void EscherTrainer::collect_regret(int traverser) {
    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        if (player == traverser) {
            // instantaneous regret r(a) = Q(a) - V̄ (acting-player frame);
            // neural RM⁺ target = max(0, γ·cum_regret(a) + r(a)).
            const float* qd = q.data_ptr<float>();
            const float* mk = mask.data_ptr<float>();
            const float* cum = lg.data_ptr<float>();  // regret net = cumulative
            double vbar = 0.0;
            for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) vbar += sig[a] * qd[a];
            auto tgt = torch::zeros({A_});
            float* td = tgt.data_ptr<float>();
            for (int a = 0; a < A_; ++a)
                if (mk[a] > 0.5f) {
                    double inst = qd[a] - vbar;
                    double v = cfg_.ncum_gamma * cum[a] + inst;
                    td[a] = v > 0.0 ? (float)v : 0.0f;
                }
            regret_smp_.push_back({obs.clone(), tgt, -1, 0.0f, 1.0f});
            return sample_uniform(mask, rng_);   // fixed b_i: uniform
        }
        return sample(sig, rng_);                // opponent ~ σ
    };
    auto finish = [&](int, float) {};
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.regret_traj, regret_, value_, act, finish);
}

void EscherTrainer::fit_regret() {
    if (regret_smp_.empty()) return;
    std::uniform_int_distribution<size_t> pick(0, regret_smp_.size() - 1);
    for (int step = 0; step < cfg_.regret_steps; ++step) {
        int B = std::min<int>(cfg_.batch_size, (int)regret_smp_.size());
        auto X = torch::empty({B, obs_dim_});
        auto Y = torch::empty({B, A_});
        for (int b = 0; b < B; ++b) {
            const auto& s = regret_smp_[pick(rng_)];
            X[b] = s.feat; Y[b] = s.target;
        }
        X = X.to(device_); Y = Y.to(device_);
        regret_opt_->zero_grad();
        auto pred = regret_->forward(X).first;   // actor logits = cumulative regret
        auto loss = torch::mse_loss(pred, Y);
        loss.backward();
        regret_opt_->step();
    }
    regret_smp_.clear();
}

// ── average phase: classification on (infoset, a_taken) under current σ ─────
void EscherTrainer::collect_avg(long t) {
    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        int a = sample(sig, rng_);
        avg_buf_->add({obs.clone(), {}, a, 0.0f, (float)t});
        return a;
    };
    auto finish = [&](int, float) {};
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.avg_traj, regret_, value_, act, finish);
}

void EscherTrainer::fit_avg() {
    auto& data = avg_buf_->data;
    if (data.empty()) return;
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    for (int step = 0; step < cfg_.avg_steps; ++step) {
        int B = std::min<int>(cfg_.batch_size, (int)data.size());
        auto X = torch::empty({B, obs_dim_});
        auto act_idx = torch::empty({B}, torch::kLong);
        auto W = torch::empty({B});
        for (int b = 0; b < B; ++b) {
            const auto& s = data[pick(rng_)];
            X[b] = s.feat; act_idx[b] = s.action; W[b] = s.weight;
        }
        X = X.to(device_); act_idx = act_idx.to(device_); W = W.to(device_);
        avg_opt_->zero_grad();
        auto logits = avg_->forward(X).first;
        auto logp = torch::log_softmax(logits, 1);
        auto nll = -logp.gather(1, act_idx.unsqueeze(1)).squeeze(1);  // [B]
        auto loss = (W * nll).mean();
        loss.backward();
        avg_opt_->step();
    }
}

void EscherTrainer::run_lbr(int iter) {
    LBRConfig lc;
    lc.num_hands = cfg_.lbr_hands;
    lc.seed = cfg_.seed + 1000 + iter;
    LBREvaluator lbr(factory_, bet_cfg_, lc, device_);
    auto res = lbr.evaluate(avg_);
    std::printf("  [iter %4d] LBR exploitability = %.4f bb/hand  (win %.3f)\n",
                iter, res.bb_per_hand, res.lbr_win_rate);
    std::fflush(stdout);
}

void EscherTrainer::save_checkpoint(int) { /* TODO */ }

// ── outer ESCHER loop ───────────────────────────────────────────────────────
void EscherTrainer::train() {
    std::printf("ESCHER HUNL: %d iters, envs=%d, value/regret/avg traj=%d/%d/%d\n",
                cfg_.iterations, kRolloutEnvs, cfg_.value_traj, cfg_.regret_traj,
                cfg_.avg_traj);
    for (iter_ = 1; iter_ <= cfg_.iterations; ++iter_) {
        auto t0 = std::chrono::steady_clock::now();
        collect_and_train_value();      // MC value under current σ
        collect_regret(0);              // traverser = seat 0
        collect_regret(1);              // traverser = seat 1
        fit_regret();                   // neural RM⁺ cumulative update
        collect_avg(iter_);             // accumulate average-policy samples
        fit_avg();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  iter %4ld   %.0f ms   (value_smp=%zu regret_smp=%zu "
                    "avg_buf=%zu)\n", iter_, ms, value_smp_.size(),
                    regret_smp_.size(), avg_buf_->data.size());
        std::fflush(stdout);
        if (iter_ % cfg_.eval_every == 0 || iter_ == cfg_.iterations)
            run_lbr((int)iter_);
    }
}

}  // namespace poker_ppo
