#include "escher_trainer.h"

#include "config.h"
#include "lbr.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>

namespace poker_ppo {

namespace {
constexpr int kRolloutEnvs = 512;   // parallel envs for batched inference
}

// ── tensor-backed Reservoir ─────────────────────────────────────────────────
EscherTrainer::Reservoir::Reservoir(size_t cap, int D, int A, uint64_t seed)
    : cap_(cap), D_(D), A_(A), rng_(seed) {
    feats_   = torch::empty({(long)cap, D}, torch::kFloat);
    targets_ = torch::empty({(long)cap, A}, torch::kFloat);
    actions_ = torch::empty({(long)cap}, torch::kLong);
    weights_ = torch::empty({(long)cap}, torch::kFloat);
}

void EscherTrainer::Reservoir::add(const float* feat, const float* target,
                                   int64_t action, float w) {
    long slot;
    if (filled_ < cap_) { slot = (long)filled_++; }
    else {
        std::uniform_int_distribution<long> d(0, n_);  // replace w/ prob cap/(n+1)
        long j = d(rng_); ++n_;
        if (j >= (long)cap_) return;                   // discard
        slot = j;
        std::memcpy(feats_.data_ptr<float>() + slot * D_, feat, D_ * sizeof(float));
        if (target) std::memcpy(targets_.data_ptr<float>() + slot * A_, target,
                                A_ * sizeof(float));
        actions_.data_ptr<long>()[slot]   = action;
        weights_.data_ptr<float>()[slot]  = w;
        return;
    }
    ++n_;
    std::memcpy(feats_.data_ptr<float>() + slot * D_, feat, D_ * sizeof(float));
    if (target) std::memcpy(targets_.data_ptr<float>() + slot * A_, target,
                            A_ * sizeof(float));
    actions_.data_ptr<long>()[slot]  = action;
    weights_.data_ptr<float>()[slot] = w;
}

EscherTrainer::Reservoir::MB
EscherTrainer::Reservoir::sample(int B, std::mt19937& rng, torch::Device dev) {
    auto idx = torch::empty({B}, torch::kLong);
    long* ip = idx.data_ptr<long>();
    std::uniform_int_distribution<long> d(0, (long)filled_ - 1);
    for (int b = 0; b < B; ++b) ip[b] = d(rng);
    return { feats_.index_select(0, idx).to(dev),
             targets_.index_select(0, idx).to(dev),
             actions_.index_select(0, idx).to(dev),
             weights_.index_select(0, idx).to(dev) };
}

void EscherTrainer::Reservoir::save(const std::string& path) const {
    long n = (long)filled_;
    torch::save(std::vector<torch::Tensor>{
        feats_.slice(0, 0, n).clone(), targets_.slice(0, 0, n).clone(),
        actions_.slice(0, 0, n).clone(), weights_.slice(0, 0, n).clone()}, path);
}

void EscherTrainer::Reservoir::load(const std::string& path) {
    std::vector<torch::Tensor> v; torch::load(v, path);
    long n = v[0].size(0);
    feats_.slice(0, 0, n).copy_(v[0]);   targets_.slice(0, 0, n).copy_(v[1]);
    actions_.slice(0, 0, n).copy_(v[2]); weights_.slice(0, 0, n).copy_(v[3]);
    filled_ = (size_t)n; n_ = n;
}

float EscherTrainer::Reservoir::target_rms() const {
    if (filled_ == 0) return 1.0f;
    return std::max(1e-6f, targets_.slice(0, 0, (long)filled_)
                              .pow(2).mean().sqrt().item<float>());
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
    value_target_ = make();
    regret_ = make();
    avg_    = make();
    { torch::NoGradGuard ng;  // value_target_ starts == value_
      auto s = value_->parameters(); auto d = value_target_->parameters();
      for (size_t i = 0; i < s.size(); ++i) d[i].copy_(s[i]); }
    value_opt_  = std::make_unique<torch::optim::Adam>(
        value_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    regret_opt_ = std::make_unique<torch::optim::Adam>(
        regret_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    avg_opt_    = std::make_unique<torch::optim::Adam>(
        avg_->parameters(), torch::optim::AdamOptions(cfg_.lr));
    avg_buf_    = std::make_unique<Reservoir>(cfg_.buf_cap, obs_dim_, A_, cfg_.seed + 7);
    regret_buf_ = std::make_unique<Reservoir>(cfg_.buf_cap, obs_dim_, A_, cfg_.seed + 8);
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
          lg = regret->actor_logits(obs_b).to(torch::kCPU);
          q  = value->critic_values(obs_b).to(torch::kCPU); }
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

// ── value phase: rollouts under current σ (both seats), buffer cleared. ──
//   MC (pure ESCHER): every visited node's target = its acting-player terminal
//     utility (high variance, only the TAKEN action gets a target).
//   DREAM bootstrap (cfg_.dream): each node's target = V̄(child) under σ — the
//     next node's expected value (frame-flipped if the opponent acts next),
//     terminal utility at leaves. One-step Bellman backup; lower variance and,
//     because V̄(child) reads the value net at the child, it covers actions the
//     MC value never sampled.
void EscherTrainer::collect_and_train_value() {
    value_smp_.clear();
    const float lam = cfg_.value_lambda;
    // full path per env (for the MC anchor); each node also gets its bootstrap
    // V̄(child) stashed in .z during traversal, then mixed at finish.
    std::vector<std::vector<std::pair<size_t, int>>> pending(kRolloutEnvs);

    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        if (lam < 1.0f && !pending[i].empty()) {
            // V̄(this node) = bootstrap target of the PREVIOUS action (frame-
            // flipped if the player changed).
            const float* qd = q.data_ptr<float>();
            double vbar = 0.0;
            for (int a = 0; a < A_; ++a) vbar += sig[a] * qd[a];
            auto [pidx, pp] = pending[i].back();
            value_smp_[pidx].z = (float)(pp == player ? vbar : -vbar);
        }
        int a = sample(sig, rng_);
        size_t si = value_smp_.size();
        value_smp_.push_back({obs.clone(), {}, a, 0.0f, 1.0f});
        pending[i].push_back({si, player});
        return a;
    };
    auto finish = [&](int i, float z_p0) {
        if (lam < 1.0f && !pending[i].empty()) {
            auto [lidx, lp] = pending[i].back();  // last node's child is terminal
            value_smp_[lidx].z = (lp == 0) ? z_p0 : -z_p0;
        }
        for (auto& [idx, p] : pending[i]) {
            float mc = (p == 0) ? z_p0 : -z_p0;            // acting-player frame
            float boot = value_smp_[idx].z;                // V̄(child), or 0 if λ=1
            value_smp_[idx].z = lam * mc + (1.0f - lam) * boot;
        }
        pending[i].clear();
    };
    // bootstrap V̄(child) reads the target net (τ>0) for deadly-triad stability.
    ActorCritic& vnet = (cfg_.value_tau > 0.f) ? value_target_ : value_;
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.value_traj, regret_, vnet, act, finish);

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
        auto qall = value_->critic_values(X);                        // [B, A]
        auto pred = qall.gather(1, act_idx.unsqueeze(1)).squeeze(1);  // Q(s,a)
        auto loss = torch::mse_loss(pred, y);
        loss.backward();
        value_opt_->step();
        if (step == cfg_.value_steps - 1) last_val_loss_ = loss.item<float>();
    }
    if (cfg_.value_tau > 0.f) {  // Polyak: target ← τ·value + (1−τ)·target
        torch::NoGradGuard ng;
        auto s = value_->parameters(); auto d = value_target_->parameters();
        for (size_t i = 0; i < s.size(); ++i)
            d[i].mul_(1.f - cfg_.value_tau).add_(s[i], cfg_.value_tau);
    }
}

// ── regret phase: fixed sampler (traverser ~ uniform, opp ~ σ), value-fn Q ──
void EscherTrainer::collect_regret(int traverser) {
    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        if (player == traverser) {
            const float* qd = q.data_ptr<float>();
            const float* mk = mask.data_ptr<float>();
            const float* cum = lg.data_ptr<float>();  // regret net output
            double vbar = 0.0;
            for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) vbar += sig[a] * qd[a];
            auto tgt = torch::zeros({A_});
            float* td = tgt.data_ptr<float>();
            if (cfg_.regret_buffer) {
                // PAPER: store the RAW instantaneous regret r(a)=Q(a)−V̄; the
                // reservoir accumulates across iters, weighted by iteration.
                for (int a = 0; a < A_; ++a)
                    if (mk[a] > 0.5f) td[a] = (float)(qd[a] - vbar);
                regret_buf_->add(obs.contiguous().data_ptr<float>(),
                                 tgt.data_ptr<float>(), -1, (float)iter_);
            } else {
                // my variant: neural RM⁺ target max(0, γ·cum(a) + r(a)).
                for (int a = 0; a < A_; ++a)
                    if (mk[a] > 0.5f) {
                        double v = cfg_.ncum_gamma * cum[a] + (qd[a] - vbar);
                        td[a] = v > 0.0 ? (float)v : 0.0f;
                    }
                regret_smp_.push_back({obs.clone(), tgt, -1, 0.0f, 1.0f});
            }
            return sample_uniform(mask, rng_);   // fixed b_i: uniform
        }
        return sample(sig, rng_);                // opponent ~ σ
    };
    auto finish = [&](int, float) {};
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.regret_traj, regret_, value_, act, finish);
}

void EscherTrainer::fit_regret() {
    if (cfg_.regret_buffer) {
        // PAPER (Deep CFR): REINIT the regret net, fit the whole reservoir
        // (tensor-backed → fast index_select minibatches), iteration-weighted
        // (linear CFR), globally normalised (RM is scale-invariant).
        if (regret_buf_->size() == 0) return;
        const auto& pc = config::kPPOConfig;
        regret_ = ActorCritic(obs_dim_, A_, pc.hidden_dim, pc.num_layers,
                              pc.hist, pc.round_summary);
        regret_->to(device_);
        regret_opt_ = std::make_unique<torch::optim::Adam>(
            regret_->parameters(), torch::optim::AdamOptions(cfg_.lr));
        float scale = regret_buf_->target_rms();
        int B = std::min(cfg_.batch_size, regret_buf_->size());
        for (int step = 0; step < cfg_.regret_steps; ++step) {
            auto mb = regret_buf_->sample(B, rng_, device_);
            auto Y = mb.target / scale;
            auto W = mb.weight.unsqueeze(1);
            regret_opt_->zero_grad();
            auto loss = (W * (regret_->actor_logits(mb.feat) - Y).pow(2)).mean();
            loss.backward();
            regret_opt_->step();
            if (step == cfg_.regret_steps - 1) {
                last_reg_loss_ = loss.item<float>();
                last_reg_mag_  = Y.abs().mean().item<float>();
            }
        }
        return;
    }
    // neural-cumulative: warm net, per-iter samples (small → per-row is fine).
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
        auto loss = torch::mse_loss(regret_->actor_logits(X), Y);
        loss.backward();
        regret_opt_->step();
        if (step == cfg_.regret_steps - 1) {
            last_reg_loss_ = loss.item<float>();
            last_reg_mag_  = Y.abs().mean().item<float>();
        }
    }
    regret_smp_.clear();
}

// ── average phase: classification on (infoset, a_taken) under current σ ─────
void EscherTrainer::collect_avg(long t) {
    auto act = [&](int i, int player, const torch::Tensor& obs,
                   const torch::Tensor& mask, torch::Tensor lg, torch::Tensor q) {
        auto sig = sigma(regret_, lg, mask);
        int a = sample(sig, rng_);
        avg_buf_->add(obs.contiguous().data_ptr<float>(), nullptr, a, (float)t);
        return a;
    };
    auto finish = [&](int, float) {};
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, kRolloutEnvs,
                cfg_.avg_traj, regret_, value_, act, finish);
}

void EscherTrainer::fit_avg() {
    if (avg_buf_->size() == 0) return;
    int B = std::min(cfg_.batch_size, avg_buf_->size());
    for (int step = 0; step < cfg_.avg_steps; ++step) {
        auto mb = avg_buf_->sample(B, rng_, device_);
        avg_opt_->zero_grad();
        auto logp = torch::log_softmax(avg_->actor_logits(mb.feat), 1);
        auto nll = -logp.gather(1, mb.action.unsqueeze(1)).squeeze(1);  // [B]
        auto loss = (mb.weight * nll).mean();
        loss.backward();
        avg_opt_->step();
    }
}

void EscherTrainer::run_lbr(int iter) {
    LBRConfig lc;
    lc.num_hands = cfg_.lbr_hands;
    lc.seed = cfg_.seed + 1000 + iter;
    LBREvaluator lbr(factory_, bet_cfg_, lc, device_);
    auto res = lbr.evaluate(avg_);                       // average policy π̄
    // Track the best LBR so the deployable strategy is the best avg, not the
    // latest (CFR's current σ oscillates; the average is what's judged).
    bool best = res.bb_per_hand < best_lbr_;
    if (best) best_lbr_ = res.bb_per_hand;
    std::printf("  [iter %4d] LBR avg=%.4f bb/hand  (win %.3f)%s\n",
                iter, res.bb_per_hand, res.lbr_win_rate, best ? "  *best*" : "");
    std::fflush(stdout);
    if (best && !cfg_.ckpt_dir.empty())   // snapshot the best avg net
        torch::save(avg_, cfg_.ckpt_dir + "/avg_best.pt");
    // cur-σ diagnostic available via LBRConfig.rm_plus on regret_ if needed.
}

// The 3 net weights fully encode the strategy: regret_ holds the cumulative
// regret, avg_ the average policy, value_ refits from current-π each iter. The
// avg reservoir is NOT saved (it refills; avg_ already encodes the average).
void EscherTrainer::save_checkpoint(int iter) {
    if (cfg_.ckpt_dir.empty()) return;
    std::filesystem::create_directories(cfg_.ckpt_dir);
    torch::save(value_,  cfg_.ckpt_dir + "/value.pt");
    torch::save(regret_, cfg_.ckpt_dir + "/regret.pt");
    torch::save(avg_,    cfg_.ckpt_dir + "/avg.pt");
    if (cfg_.regret_buffer)  // the reservoir IS the cumulative-regret state
        regret_buf_->save(cfg_.ckpt_dir + "/regret_buf.pt");
    std::ofstream(cfg_.ckpt_dir + "/iter.txt") << iter;
    std::printf("  [ckpt] saved iter %d -> %s\n", iter, cfg_.ckpt_dir.c_str());
}

bool EscherTrainer::try_resume() {
    if (cfg_.ckpt_dir.empty() ||
        !std::filesystem::exists(cfg_.ckpt_dir + "/avg.pt"))
        return false;
    torch::load(value_,  cfg_.ckpt_dir + "/value.pt");
    torch::load(regret_, cfg_.ckpt_dir + "/regret.pt");
    torch::load(avg_,    cfg_.ckpt_dir + "/avg.pt");
    value_->to(device_); regret_->to(device_); avg_->to(device_);
    { torch::NoGradGuard ng;  // resync target net to the loaded value net
      auto s = value_->parameters(); auto d = value_target_->parameters();
      for (size_t i = 0; i < s.size(); ++i) d[i].copy_(s[i]); }
    if (cfg_.regret_buffer &&
        std::filesystem::exists(cfg_.ckpt_dir + "/regret_buf.pt"))
        regret_buf_->load(cfg_.ckpt_dir + "/regret_buf.pt");
    std::ifstream(cfg_.ckpt_dir + "/iter.txt") >> resume_iter_;
    std::printf("  [resume] loaded checkpoint at iter %d\n", resume_iter_);
    return true;
}

// ── outer ESCHER loop ───────────────────────────────────────────────────────
void EscherTrainer::train() {
    std::printf("ESCHER HUNL: %d iters, envs=%d, value/regret/avg traj=%d/%d/%d\n",
                cfg_.iterations, kRolloutEnvs, cfg_.value_traj, cfg_.regret_traj,
                cfg_.avg_traj);
    try_resume();
    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto el = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count(); };
    for (iter_ = resume_iter_ + 1; iter_ <= cfg_.iterations; ++iter_) {
        auto t0 = clk(); collect_and_train_value();
        auto t1 = clk(); collect_regret(0); collect_regret(1);
        auto t2 = clk(); fit_regret();
        auto t3 = clk();
        if (iter_ > cfg_.avg_warmup) collect_avg(iter_);  // burn-in: drop early σ
        auto t4 = clk(); fit_avg();
        auto t5 = clk();
        std::printf("  iter %4ld  %.0fms [val %.0f reg-roll %.0f reg-fit %.0f "
                    "avg-roll %.0f avg-fit %.0f]  vloss=%.3f rloss=%.4f "
                    "rmag=%.3f\n", iter_, el(t0, t5), el(t0, t1), el(t1, t2),
                    el(t2, t3), el(t3, t4), el(t4, t5), last_val_loss_,
                    last_reg_loss_, last_reg_mag_);
        std::fflush(stdout);
        if (!cfg_.ckpt_dir.empty() && iter_ % cfg_.ckpt_every == 0)
            save_checkpoint((int)iter_);
        if (iter_ % cfg_.eval_every == 0 || iter_ == cfg_.iterations)
            run_lbr((int)iter_);
    }
    save_checkpoint((int)cfg_.iterations);
}

}  // namespace poker_ppo
