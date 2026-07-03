#include "escher_trainer.h"

#include "config.h"
#include "lbr.h"

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <thread>

namespace poker_ppo {

namespace {
// parallel rollout envs: EscherConfig::rollout_envs (ESCHER_ENVS)

// RAII bf16 autocast for the SGD fits: matmuls run in bf16 (tensor cores +
// half the activation bandwidth) while grads/params stay fp32 — bf16 keeps
// fp32's exponent range so no GradScaler is needed. Only the forward+loss is
// wrapped; backward/step run in fp32. POKER_PPO_ESCHER_NO_AMP=1 disables.
struct AutocastBF16 {
    bool on_;
    explicit AutocastBF16(torch::Device dev)
        : on_(dev.is_cuda() && std::getenv("POKER_PPO_ESCHER_NO_AMP") == nullptr) {
        if (!on_) return;
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    }
    ~AutocastBF16() {
        if (!on_) return;
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::clear_cache();
    }
};

// Params of `net` excluding the head it never trains (regret/avg use only the
// actor via actor_logits, value only the critic via critic_values) + the shared
// encoder. ForeachAdam requires EVERY managed param to receive a gradient (stock
// Adam silently skips grad-less ones); the unused head would break its step.
std::vector<torch::Tensor> params_except(ActorCritic& net, const char* prefix) {
    std::vector<torch::Tensor> out;
    for (const auto& item : net->named_parameters()) {
        const std::string& k = item.key();
        if (k.rfind(prefix, 0) == 0) continue;   // name starts with prefix → skip
        out.push_back(item.value());
    }
    return out;
}
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

EscherTrainer::Reservoir::MB
EscherTrainer::Reservoir::upload(torch::Device dev) const {
    long n = (long)filled_;
    return { feats_.slice(0, 0, n).to(dev),
             targets_.slice(0, 0, n).to(dev),
             actions_.slice(0, 0, n).to(dev),
             weights_.slice(0, 0, n).to(dev) };
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

// ── CUDA-graphed fit step state ─────────────────────────────────────────────
// One captured fwd+bwd graph per net (value/regret/avg). Static input tensors
// are refilled by on-device index_select each step; replay() recomputes loss
// + grads into stable tensors that ForeachAdam::step(grads) consumes. Same
// recipe as ppo.cpp's ensure_update_graph. Paper-mode's per-iter regret
// reinit stays graph-safe because reinit_regret_inplace() copies fresh
// weights into the SAME parameter tensors the capture recorded.
struct EscherTrainer::FitGraphs {
    struct One {
        int  B = 0;
        bool ready = false, failed = false;
        torch::Tensor feat, target, action, weight;   // static inputs (device)
        torch::Tensor loss;                            // static output scalar
        std::vector<torch::Tensor> grads;              // static grad outputs
        std::unique_ptr<at::cuda::CUDAGraph> graph;
    } value, regret, avg;
    One& of(FitKind k) {
        switch (k) {
        case FitKind::Value:  return value;
        case FitKind::Regret: return regret;
        default:              return avg;
        }
    }
};

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
    // NOTE order: value_/regret_/avg_ FIRST (identical init RNG draws to the
    // pre-value_target_ code at 5c29560), then the auxiliary nets — so adding
    // value_target_/regret_ema_ does NOT shift the core nets' init basin. (The
    // earlier ordering silently moved the σ oscillation into a worse basin.)
    value_  = make();
    regret_ = make();
    avg_    = make();
    value_target_ = make();
    regret_ema_ = make();
    { torch::NoGradGuard ng;  // value_target_ starts == value_; ema starts == regret_
      auto s = value_->parameters(); auto d = value_target_->parameters();
      for (size_t i = 0; i < s.size(); ++i) d[i].copy_(s[i]);
      auto rs = regret_->parameters(); auto rd = regret_ema_->parameters();
      for (size_t i = 0; i < rs.size(); ++i) rd[i].copy_(rs[i]);
      auto rb = regret_->buffers(); auto rdb = regret_ema_->buffers();
      for (size_t i = 0; i < rb.size(); ++i) rdb[i].copy_(rb[i]); }
    value_opt_  = std::make_unique<ForeachAdam>(params_except(value_,  "actor."),  cfg_.lr);
    regret_opt_ = std::make_unique<ForeachAdam>(params_except(regret_, "critic."), cfg_.lr);
    avg_opt_    = std::make_unique<ForeachAdam>(params_except(avg_,    "critic."), cfg_.lr);
    avg_buf_    = std::make_unique<Reservoir>(cfg_.buf_cap, obs_dim_, A_, cfg_.seed + 7);
    regret_buf_ = std::make_unique<Reservoir>(cfg_.buf_cap, obs_dim_, A_, cfg_.seed + 8);
    if (cfg_.value_buf_cap > 0)
        value_buf_ = std::make_unique<Reservoir>(cfg_.value_buf_cap, obs_dim_, A_,
                                                 cfg_.seed + 9);
    // Per-iteration staging (cleared each iter). Caps sized ~16 decision
    // nodes/trajectory; overflow reservoir-subsamples uniformly (benign).
    if (cfg_.value_buf_cap <= 0)
        value_iter_buf_ = std::make_unique<Reservoir>(
            std::max(1 << 16, cfg_.value_traj * 16), obs_dim_, A_, cfg_.seed + 10);
    if (!cfg_.regret_buffer)
        regret_iter_buf_ = std::make_unique<Reservoir>(
            std::max(1 << 16, cfg_.regret_traj * 16), obs_dim_, A_, cfg_.seed + 11);
    graphs_ = std::make_unique<FitGraphs>();
}

EscherTrainer::~EscherTrainer() = default;

// ── σ = RM⁺ on the regret net's actor logits, masked to legal ───────────────
std::vector<float> EscherTrainer::sigma(const float* lg, const float* mk) {
    // lg, mk are CPU rows [A]
    std::vector<float> p(A_, 0.0f);
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

// ── Predictive/Optimistic RM⁺: σ = RM⁺(cum + η·(Q − V̄_base)) ────────────────
// V̄_base = Σ σ_base·Q with σ_base = RM⁺(cum). The current instantaneous regret
// Q−V̄_base is an optimistic prediction of the next regret; adding it to the
// cumulative before RM⁺ is exactly one step of lookahead (η=1 = full step).
std::vector<float> EscherTrainer::sigma_pred(const float* lg, const float* qd,
                                             const float* mk, float eta) {
    auto base = sigma(lg, mk);   // σ_base = RM⁺(cum)
    double vbar = 0.0;
    for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) vbar += base[a] * qd[a];
    std::vector<float> p(A_, 0.0f);
    double s = 0.0;
    for (int a = 0; a < A_; ++a)
        if (mk[a] > 0.5f) {                          // RM⁺(cum + η·(Q − V̄))
            double pl = static_cast<double>(lg[a]) + eta * (qd[a] - vbar);
            double v = pl > 0.0 ? pl : 0.0;
            p[a] = static_cast<float>(v); s += v;
        }
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

int EscherTrainer::sample_uniform(const float* mk, std::mt19937& rng) {
    std::vector<int> legal;
    for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) legal.push_back(a);
    std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
    return legal[d(rng)];
}

// ── batched rollout engine ──────────────────────────────────────────────────
// Runs kRolloutEnvs envs until `n_traj` terminations. Each step: batch the
// acting obs, forward regret_ (σ) and — only when a consumer needs Q —
// value_, then per-env `act(...)` chooses the action and records;
// `finish(i, z_p0)` fires on terminal.
//
// Each env's obs/mask lives in row i of one contiguous CPU tensor pair
// (written in place by step_into/reset_into), so batching is one
// index_select instead of per-row tensor assignments, and the hot loop
// hands raw row POINTERS to the callbacks — no per-row tensor views.
// act(i, player, obs_row, mask_row, lg_row, q_row, cum_row); q_row/cum_row
// are nullptr when that forward is skipped.
namespace {
struct EnvSlot {
    std::unique_ptr<IPokerEnvironment> env;
    int player = 0;
    float z = 0.0f;            // accumulated P0-frame reward this trajectory
    bool live = false;
};
}  // namespace

template <class Act, class Finish>
static void run_rollout(IPokerEnvironmentFactory& factory, const BetConfig& bet,
                        int obs_dim, int A, torch::Device device, int n_envs,
                        int n_traj, ActorCritic& regret, ActorCritic& value,
                        bool need_q, Act&& act, Finish&& finish,
                        ActorCritic* cum_net = nullptr) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    auto obs_all  = torch::empty({n_envs, obs_dim}, opts);
    auto mask_all = torch::empty({n_envs, A}, opts);
    float* ob = obs_all.data_ptr<float>();
    float* mb = mask_all.data_ptr<float>();
    std::vector<EnvSlot> slots(n_envs);
    for (int i = 0; i < n_envs; ++i) {
        auto& s = slots[i];
        s.env = factory.create(bet);
        s.env->reset_into(ob + (size_t)i * obs_dim, mb + (size_t)i * A);
        s.player = s.env->current_player(); s.z = 0.f; s.live = true;
    }
    int finished = 0;
    auto idx_t = torch::empty({n_envs}, torch::kLong);
    while (finished < n_traj) {
        long* ip = idx_t.data_ptr<long>();
        int n_live = 0;
        for (int i = 0; i < n_envs; ++i) if (slots[i].live) ip[n_live++] = i;
        if (n_live == 0) break;
        torch::Tensor obs_b =
            (n_live == n_envs)
                ? obs_all.to(device)
                : obs_all.index_select(0, idx_t.slice(0, 0, n_live)).to(device);
        torch::Tensor lg, q, cg;
        { torch::NoGradGuard ng;
          lg = regret->actor_logits(obs_b).to(torch::kCPU).contiguous();
          if (need_q)
              q = value->critic_values(obs_b).to(torch::kCPU).contiguous();
          if (cum_net)
              cg = (*cum_net)->actor_logits(obs_b).to(torch::kCPU).contiguous(); }
        const float* lgp = lg.data_ptr<float>();
        const float* qp  = q.defined()  ? q.data_ptr<float>()  : nullptr;
        const float* cgp = cg.defined() ? cg.data_ptr<float>() : nullptr;
        for (int k = 0; k < n_live; ++k) {
            int i = (int)ip[k];
            auto& s = slots[i];
            int a = act(i, s.player,
                        ob + (size_t)i * obs_dim, mb + (size_t)i * A,
                        lgp + (size_t)k * A,
                        qp  ? qp  + (size_t)k * A : nullptr,
                        cgp ? cgp + (size_t)k * A : nullptr);
            auto r = s.env->step_into(a, ob + (size_t)i * obs_dim,
                                      mb + (size_t)i * A);
            s.z += r.reward;
            if (r.done) {
                finish(i, s.z);
                ++finished;
                if (finished >= n_traj) { s.live = false; continue; }
                s.env->reset_into(ob + (size_t)i * obs_dim,
                                  mb + (size_t)i * A);
                s.player = s.env->current_player(); s.z = 0.f;
            } else {
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
void EscherTrainer::collect_and_train_value(bool fit) {
    const float lam = cfg_.value_lambda;
    // Q is only consumed here when the bootstrap or predictive σ needs it —
    // pure-ESCHER (λ=1, no predictive) skips the value forward entirely.
    const bool need_q = lam < 1.0f || cfg_.predictive > 0.f;
    Reservoir& vbuf = value_buf_ ? *value_buf_ : *value_iter_buf_;
    if (!value_buf_) value_iter_buf_->clear();

    // Per-env trajectory staging: nodes are held (feature row + action/player/
    // boot) until the trajectory terminates and the MC anchor z is known, then
    // flushed into the reservoir. Bounded at kMaxNodes (HUNL hands are far
    // shorter); overflow nodes are dropped.
    constexpr int kMaxNodes = 64;
    struct Node { int64_t action; int player; float boot; };
    std::vector<float> stage_feat((size_t)cfg_.rollout_envs * kMaxNodes * obs_dim_);
    std::vector<Node>  stage_node((size_t)cfg_.rollout_envs * kMaxNodes);
    std::vector<int>   n_stage(cfg_.rollout_envs, 0);

    auto act = [&](int i, int player, const float* obs, const float* mk,
                   const float* lg, const float* qd, const float* /*cum*/) {
        auto sig = cfg_.predictive > 0.f
                       ? sigma_pred(lg, qd, mk, cfg_.predictive)
                       : sigma(lg, mk);
        if (lam < 1.0f && n_stage[i] > 0) {
            // V̄(this node) = bootstrap target of the PREVIOUS action (frame-
            // flipped if the player changed).
            double vbar = 0.0;
            for (int a = 0; a < A_; ++a) vbar += sig[a] * qd[a];
            Node& prev = stage_node[(size_t)i * kMaxNodes + n_stage[i] - 1];
            prev.boot = (float)(prev.player == player ? vbar : -vbar);
        }
        int a;
        if (cfg_.value_eps > 0.f) {
            // ε-uniform exploration for the SAMPLED action only (coverage);
            // the bootstrap V̄ above still uses the on-policy σ, so only Q's
            // estimated continuation carries the O(ε) bias, not the regret's
            // target weights.
            std::vector<float> samp = sig;
            int nl = 0; for (int j = 0; j < A_; ++j) if (mk[j] > 0.5f) ++nl;
            const float e = cfg_.value_eps;
            for (int j = 0; j < A_; ++j)
                if (mk[j] > 0.5f) samp[j] = (1.f - e) * sig[j] + e / nl;
            a = sample(samp, rng_);
        } else {
            a = sample(sig, rng_);
        }
        if (n_stage[i] < kMaxNodes) {
            int k = n_stage[i]++;
            std::memcpy(&stage_feat[((size_t)i * kMaxNodes + k) * obs_dim_],
                        obs, obs_dim_ * sizeof(float));
            stage_node[(size_t)i * kMaxNodes + k] = {a, player, 0.0f};
        }
        return a;
    };
    std::vector<float> vtgt(A_, 0.f);   // scalar z parked at index 0 for the buf
    auto finish = [&](int i, float z_p0) {
        int n = n_stage[i];
        if (lam < 1.0f && n > 0) {       // last node's child is terminal
            Node& last = stage_node[(size_t)i * kMaxNodes + n - 1];
            last.boot = (last.player == 0) ? z_p0 : -z_p0;
        }
        for (int k = 0; k < n; ++k) {
            const Node& nd = stage_node[(size_t)i * kMaxNodes + k];
            float mc = (nd.player == 0) ? z_p0 : -z_p0;    // acting-player frame
            vtgt[0] = lam * mc + (1.0f - lam) * nd.boot;
            vbuf.add(&stage_feat[((size_t)i * kMaxNodes + k) * obs_dim_],
                     vtgt.data(), nd.action,
                     value_buf_ ? (float)iter_ : 1.0f);    // recency weight
        }
        n_stage[i] = 0;
    };
    // bootstrap V̄(child) reads the target net (τ>0) for deadly-triad stability.
    ActorCritic& vnet = (cfg_.value_tau > 0.f) ? value_target_ : value_;
    ActorCritic& snet = (cfg_.regret_ema > 0.f) ? regret_ema_ : regret_;  // smoothed σ
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, cfg_.rollout_envs,
                cfg_.value_traj, snet, vnet, need_q, act, finish);
    if (!fit) return;   // resume prefill: data only

    // train Q(obs, a_taken) -> signed terminal utility (acting-player frame).
    // value_buf_ path: reservoir across iters, recency-weighted (~window× the
    // data). Else: this iter's samples only (weights ≡ 1).
    // Fit budget ∝ reservoir fullness: while the buffer refills (fresh start
    // or resume — it isn't checkpointed), a full-budget fit overfits the few
    // rows present and shocks Q, whose error the regret cumulative then
    // rectifies in (the observed post-resume rmag inflation).
    int vsteps = cfg_.value_steps;
    if (value_buf_)
        vsteps = static_cast<int>(std::lround(
            cfg_.value_steps *
            std::min(1.0, vbuf.size() / (double)cfg_.value_buf_cap)));
    last_val_loss_ = run_fit(FitKind::Value, value_, *value_opt_, vbuf,
                             vsteps, /*scale_by_rms=*/false);

    if (cfg_.value_tau > 0.f) {  // Polyak: target ← τ·value + (1−τ)·target
        torch::NoGradGuard ng;
        auto s = value_->parameters(); auto d = value_target_->parameters();
        for (size_t i = 0; i < s.size(); ++i)
            d[i].mul_(1.f - cfg_.value_tau).add_(s[i], cfg_.value_tau);
    }
}

// ── regret phase: fixed sampler (traverser ~ uniform, opp ~ σ), value-fn Q ──
void EscherTrainer::collect_regret(int traverser) {
    std::vector<float> tgt(A_);          // reused target row (no per-node alloc)
    auto act = [&](int i, int player, const float* obs, const float* mk,
                   const float* lg, const float* qd, const float* cum_row) {
        auto sig = cfg_.predictive > 0.f
                       ? sigma_pred(lg, qd, mk, cfg_.predictive)
                       : sigma(lg, mk);
        if (player == traverser) {
            // cum = TRUE cumulative (regret_); lg may be the smoothed EMA σ-net,
            // so take cum from cum_row when the EMA path supplies it.
            const float* cum = cum_row ? cum_row : lg;
            double vbar = 0.0;
            for (int a = 0; a < A_; ++a) if (mk[a] > 0.5f) vbar += sig[a] * qd[a];
            std::fill(tgt.begin(), tgt.end(), 0.0f);
            if (cfg_.regret_buffer) {
                // PAPER: store the RAW instantaneous regret r(a)=Q(a)−V̄; the
                // reservoir accumulates across iters, weighted by iteration.
                for (int a = 0; a < A_; ++a)
                    if (mk[a] > 0.5f) tgt[a] = (float)(qd[a] - vbar);
                regret_buf_->add(obs, tgt.data(), -1, (float)iter_);
            } else {
                // my variant: neural RM⁺ target max(0, γ·cum(a) + r(a)).
                for (int a = 0; a < A_; ++a)
                    if (mk[a] > 0.5f) {
                        double v = cfg_.ncum_gamma * cum[a] + (qd[a] - vbar);
                        tgt[a] = v > 0.0 ? (float)v : 0.0f;
                    }
                regret_iter_buf_->add(obs, tgt.data(), -1, 1.0f);
            }
            return sample_uniform(mk, rng_);     // fixed b_i: uniform
        }
        return sample(sig, rng_);                // opponent ~ σ
    };
    auto finish = [&](int, float) {};
    // Play the smoothed σ (regret_ema_) but feed the TRUE regret_ as cum_net so
    // the neural-cum cumulative keeps accumulating unsmoothed.
    ActorCritic& snet = (cfg_.regret_ema > 0.f) ? regret_ema_ : regret_;
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, cfg_.rollout_envs,
                cfg_.regret_traj, snet, value_, /*need_q=*/true, act, finish,
                cfg_.regret_ema > 0.f ? &regret_ : nullptr);
}

// PAPER (Deep CFR) reinit, graph-safe: fresh weights are copied INTO the
// existing parameter/buffer tensors (stable addresses for the captured fit
// graph); the optimizer is rebuilt over the same tensors (fresh Adam state,
// exactly the paper's re-train-from-scratch semantics). The fresh net is
// drawn from the same ctor, so the init RNG stream matches the old
// construct-and-swap code.
void EscherTrainer::reinit_regret_inplace() {
    const auto& pc = config::kPPOConfig;
    ActorCritic fresh(obs_dim_, A_, pc.hidden_dim, pc.num_layers,
                      pc.hist, pc.round_summary);
    fresh->to(device_);
    torch::NoGradGuard ng;
    auto s = fresh->parameters(); auto d = regret_->parameters();
    for (size_t i = 0; i < s.size(); ++i) d[i].copy_(s[i]);
    auto sb = fresh->buffers(); auto db = regret_->buffers();
    for (size_t i = 0; i < sb.size(); ++i) db[i].copy_(sb[i]);
    regret_opt_ = std::make_unique<ForeachAdam>(
        params_except(regret_, "critic."), cfg_.lr);
}

void EscherTrainer::fit_regret() {
    if (cfg_.regret_buffer) {
        // PAPER (Deep CFR): REINIT the regret net, fit the whole reservoir,
        // iteration-weighted (linear CFR), globally normalised (RM is
        // scale-invariant).
        if (regret_buf_->size() == 0) return;
        reinit_regret_inplace();
        last_reg_loss_ = run_fit(FitKind::Regret, regret_, *regret_opt_,
                                 *regret_buf_, cfg_.regret_steps,
                                 /*scale_by_rms=*/true, &last_reg_mag_);
        return;
    }
    // neural-cumulative: warm net, this iter's samples only (weights ≡ 1).
    if (regret_iter_buf_->size() == 0) return;
    last_reg_loss_ = run_fit(FitKind::Regret, regret_, *regret_opt_,
                             *regret_iter_buf_, cfg_.regret_steps,
                             /*scale_by_rms=*/false, &last_reg_mag_);
    regret_iter_buf_->clear();
}

// ── average phase: classification on (infoset, a_taken) under current σ ─────
void EscherTrainer::collect_avg(long t) {
    // Q is only consumed by the predictive-σ variant.
    const bool need_q = cfg_.predictive > 0.f;
    // t^k iteration weight (k=1 linear CFR; see EscherConfig::avg_pow).
    const float w_avg = std::pow(static_cast<float>(t), cfg_.avg_pow);
    auto act = [&](int i, int player, const float* obs, const float* mk,
                   const float* lg, const float* qd, const float* /*cum*/) {
        auto sig = cfg_.predictive > 0.f
                       ? sigma_pred(lg, qd, mk, cfg_.predictive)
                       : sigma(lg, mk);
        int a = sample(sig, rng_);
        avg_buf_->add(obs, nullptr, a, w_avg);
        return a;
    };
    auto finish = [&](int, float) {};
    ActorCritic& snet = (cfg_.regret_ema > 0.f) ? regret_ema_ : regret_;  // smoothed σ
    run_rollout(factory_, bet_cfg_, obs_dim_, A_, device_, cfg_.rollout_envs,
                cfg_.avg_traj, snet, value_, need_q, act, finish);
}

void EscherTrainer::fit_avg() {
    if (avg_buf_->size() == 0) return;
    // Budget ∝ fullness, as in the value fit: after a resume the (warm, good)
    // avg net would otherwise be yanked by full-budget fits on a near-empty
    // refilling reservoir.
    const int steps = static_cast<int>(std::lround(
        cfg_.avg_steps *
        std::min(1.0, avg_buf_->size() / (double)cfg_.buf_cap)));
    run_fit(FitKind::Avg, avg_, *avg_opt_, *avg_buf_, steps,
            /*scale_by_rms=*/false);
}

// ── unified fit engine ──────────────────────────────────────────────────────
// CUDA path: ONE bulk upload of the filled reservoir rows per fit, then every
// minibatch is on-device randint + index_select into the captured graph's
// static inputs → replay (fwd+bwd) → ForeachAdam::step(static grads). No CPU
// gather and no host sync inside the step loop (one .item() at the end), so
// the whole fit pipelines on the GPU. Eager fallbacks: device-resident loop
// (partial batch / capture failure / POKER_PPO_ESCHER_NO_FIT_GRAPH=1), and
// the old Reservoir::sample loop on CPU devices.
float EscherTrainer::run_fit(FitKind kind, ActorCritic& net, ForeachAdam& opt,
                             Reservoir& buf, int steps, bool scale_by_rms,
                             float* out_target_mag) {
    if (buf.size() == 0 || steps <= 0) return 0.f;
    const int n = buf.size();
    const int B = std::min(cfg_.batch_size, n);

    // Loss shapes (identical math to the pre-engine per-fit loops):
    //   Regret: iteration-weighted MSE on the full-A regret row.
    //   Avg:    iteration-weighted NLL of the taken action.
    //   Value:  recency-normalised weighted MSE on Q(s, a_taken) vs z at [0].
    auto loss_of = [&](const torch::Tensor& feat, const torch::Tensor& target,
                       const torch::Tensor& action, const torch::Tensor& weight)
        -> torch::Tensor {
        switch (kind) {
        case FitKind::Regret:
            return (weight.unsqueeze(1) *
                    (net->actor_logits(feat) - target).pow(2)).mean();
        case FitKind::Avg: {
            auto logp = torch::log_softmax(net->actor_logits(feat), 1);
            auto nll  = -logp.gather(1, action.unsqueeze(1)).squeeze(1);
            return (weight * nll).mean();
        }
        default: {  // Value
            auto qall = net->critic_values(feat);                    // [B, A]
            auto pred = qall.gather(1, action.unsqueeze(1)).squeeze(1);
            auto y    = target.select(1, 0);                         // z at [0]
            auto W    = weight / weight.mean().clamp_min(1e-6f);
            return (W * (pred - y).pow(2)).mean();
        }
        }
    };

    if (!device_.is_cuda()) {   // CPU device: old per-step gather loop
        const float scale = scale_by_rms ? buf.target_rms() : 1.0f;
        torch::Tensor last_loss, last_y;
        for (int step = 0; step < steps; ++step) {
            auto mb = buf.sample(B, rng_, device_);
            auto Y  = scale != 1.0f ? mb.target / scale : mb.target;
            opt.zero_grad();
            auto loss = loss_of(mb.feat, Y, mb.action, mb.weight);
            loss.backward();
            opt.step();
            if (step == steps - 1) { last_loss = loss; last_y = Y; }
        }
        if (out_target_mag) *out_target_mag = last_y.abs().mean().item<float>();
        return last_loss.item<float>();
    }

    auto dv = buf.upload(device_);                       // one bulk H2D
    const float scale = scale_by_rms
        ? std::max(1e-6f, dv.target.pow(2).mean().sqrt().item<float>())
        : 1.0f;
    const bool use_amp = std::getenv("POKER_PPO_ESCHER_NO_AMP") == nullptr;
    auto i_dev = torch::TensorOptions().dtype(torch::kLong).device(device_);

    // one-time capture, full batches only (partial batch → eager path)
    auto& fg = graphs_->of(kind);
    if (!fg.ready && !fg.failed && B == cfg_.batch_size) {
        if (std::getenv("POKER_PPO_ESCHER_NO_FIT_GRAPH") != nullptr) {
            fg.failed = true;
        } else {
            try {
                const long D = dv.feat.size(1);
                auto f_dev = torch::TensorOptions().dtype(torch::kFloat)
                                 .device(device_);
                fg.feat   = torch::zeros({B, D}, f_dev);
                fg.target = torch::zeros({B, (long)A_}, f_dev);
                fg.action = torch::zeros({B}, i_dev);
                fg.weight = torch::ones({B}, f_dev);
                // live minibatch for warmup + capture
                auto idx0 = torch::randint(0, n, {B}, i_dev);
                at::index_select_out(fg.feat,   dv.feat,   0, idx0);
                at::index_select_out(fg.target, dv.target, 0, idx0);
                at::index_select_out(fg.action, dv.action, 0, idx0);
                at::index_select_out(fg.weight, dv.weight, 0, idx0);
                if (scale != 1.0f) fg.target.div_(scale);

                // Autocast cache must be OFF from warmup through capture: a
                // cached bf16 weight cast would be baked into the graph as a
                // stale constant; uncached casts are recorded as kernels that
                // re-read the live fp32 weights on every replay.
                if (use_amp) at::autocast::set_autocast_cache_enabled(false);
                auto run_once = [&] {
                    if (use_amp) {
                        at::autocast::set_autocast_enabled(at::kCUDA, true);
                        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
                    }
                    auto l = loss_of(fg.feat, fg.target, fg.action, fg.weight);
                    if (use_amp)
                        at::autocast::set_autocast_enabled(at::kCUDA, false);
                    return l;
                };
                const auto& params = opt.params();
                torch::autograd::variable_list plist(params.begin(),
                                                     params.end());
                auto stream = at::cuda::getStreamFromPool();
                {
                    c10::cuda::CUDAStreamGuard guard(stream);
                    for (int w = 0; w < 3; ++w) {   // warmup off-graph
                        auto l = run_once();
                        (void)torch::autograd::grad({l}, plist);
                    }
                    at::cuda::getCurrentCUDAStream().synchronize();
                    fg.graph = std::make_unique<at::cuda::CUDAGraph>();
                    fg.graph->capture_begin();
                    auto l = run_once();
                    auto g = torch::autograd::grad({l}, plist);
                    fg.graph->capture_end();
                    fg.loss = l;
                    fg.grads.assign(g.begin(), g.end());
                }
                if (use_amp) at::autocast::set_autocast_cache_enabled(true);
                torch::cuda::synchronize();
                fg.B = B; fg.ready = true;
                std::printf("  [fit] CUDA-graphed fit step, kind=%d B=%d "
                            "(POKER_PPO_ESCHER_NO_FIT_GRAPH=1 disables)\n",
                            (int)kind, B);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  [fit] graph capture failed, eager "
                             "fallback: %s\n", e.what());
                at::autocast::set_autocast_enabled(at::kCUDA, false);
                at::autocast::set_autocast_cache_enabled(true);
                fg.graph.reset(); fg.grads.clear();
                fg.feat = fg.target = fg.action = fg.weight = fg.loss
                        = torch::Tensor();
                fg.failed = true;
            }
        }
    }
    const bool graphed = fg.ready && fg.B == B;

    torch::Tensor last_loss, last_y;
    for (int step = 0; step < steps; ++step) {
        auto idx = torch::randint(0, n, {B}, i_dev);
        if (graphed) {
            at::index_select_out(fg.feat,   dv.feat,   0, idx);
            at::index_select_out(fg.target, dv.target, 0, idx);
            at::index_select_out(fg.action, dv.action, 0, idx);
            at::index_select_out(fg.weight, dv.weight, 0, idx);
            if (scale != 1.0f) fg.target.div_(scale);
            fg.graph->replay();                 // fwd+bwd → fg.loss, fg.grads
            opt.step(fg.grads);
        } else {
            auto feat = dv.feat.index_select(0, idx);
            auto tgt  = dv.target.index_select(0, idx);
            if (scale != 1.0f) tgt = tgt / scale;
            auto actn = dv.action.index_select(0, idx);
            auto w    = dv.weight.index_select(0, idx);
            opt.zero_grad();
            torch::Tensor loss;
            { AutocastBF16 ac(device_);
              loss = loss_of(feat, tgt, actn, w); }
            loss.backward();
            opt.step();
            if (step == steps - 1) { last_loss = loss; last_y = tgt; }
        }
    }
    // single host sync per fit
    if (graphed) {
        if (out_target_mag)
            *out_target_mag = fg.target.abs().mean().item<float>();
        return fg.loss.item<float>();
    }
    if (out_target_mag) *out_target_mag = last_y.abs().mean().item<float>();
    return last_loss.item<float>();
}

void EscherTrainer::run_lbr(int iter) {
    if (iter <= cfg_.avg_warmup) {
        // collect_avg hasn't run yet — the avg net is still random init, so
        // an eval here measures nothing (and its LBR would poison best_lbr_).
        std::printf("  [iter %4d] LBR skipped (avg warmup ends at %d)\n",
                    iter, cfg_.avg_warmup);
        std::fflush(stdout);
        return;
    }
    // Shard the hands across lbr_threads evaluators (own env + RNG stream
    // each; hands are independent, so the merged bound is the same
    // estimator). Shard 0 prints the attribution table for its subset.
    auto eval_sharded = [&](ActorCritic& net, bool rm_plus,
                            uint64_t seed) -> LBREvaluator::Result {
        const int T = std::max(1, std::min(cfg_.lbr_threads, cfg_.lbr_hands));
        std::vector<LBREvaluator::Result> shard(T);
        auto run_shard = [&](int t) {
            LBRConfig lc;
            lc.num_hands  = cfg_.lbr_hands / T
                            + (t == 0 ? cfg_.lbr_hands % T : 0);
            lc.seed       = seed + 7919u * static_cast<uint64_t>(t);
            lc.rm_plus    = rm_plus;
            lc.print_diag = (t == 0);
            LBREvaluator ev(factory_, bet_cfg_, lc, device_);
            shard[t] = ev.evaluate(net);
        };
        if (T == 1) { run_shard(0); return shard[0]; }
        std::vector<std::thread> ws;
        ws.reserve(T);
        for (int t = 0; t < T; ++t) ws.emplace_back(run_shard, t);
        for (auto& w : ws) w.join();
        LBREvaluator::Result r;
        double bb = 0.0, mbb = 0.0, win = 0.0;
        for (const auto& s : shard) {
            r.num_hands += s.num_hands;
            bb  += s.bb_per_hand  * s.num_hands;
            mbb += s.mbb_per_hand * s.num_hands;
            win += s.lbr_win_rate * s.num_hands;
            r.wall_ms = std::max(r.wall_ms, s.wall_ms);
        }
        const double n = std::max(1, r.num_hands);
        r.bb_per_hand  = bb / n;
        r.mbb_per_hand = mbb / n;
        r.lbr_win_rate = win / n;
        return r;
    };

    auto res = eval_sharded(avg_, /*rm_plus=*/false,   // average policy π̄
                            cfg_.seed + 1000 + iter);
    // Track the best LBR so the deployable strategy is the best avg, not the
    // latest (CFR's current σ oscillates; the average is what's judged).
    bool best = res.bb_per_hand < best_lbr_;
    if (best) best_lbr_ = res.bb_per_hand;
    std::printf("  [iter %4d] LBR avg=%.4f bb/hand  (win %.3f, %.0fs)%s\n",
                iter, res.bb_per_hand, res.lbr_win_rate,
                res.wall_ms / 1000.0, best ? "  *best*" : "");
    std::fflush(stdout);
    if (best && !cfg_.ckpt_dir.empty()) {  // snapshot the best avg net
        // First best can precede the first save_checkpoint() (iter ckpt_every),
        // which is what otherwise creates the directory.
        std::filesystem::create_directories(cfg_.ckpt_dir);
        torch::save(avg_, cfg_.ckpt_dir + "/avg_best.pt");
    }
    if (cfg_.lbr_cur) {   // drift-vs-floor diagnostic: attack the played σ
        ActorCritic& played = (cfg_.regret_ema > 0.f) ? regret_ema_ : regret_;
        auto cres = eval_sharded(played, /*rm_plus=*/true,
                                 cfg_.seed + 2000 + iter);
        bool cbest = cres.bb_per_hand < best_cur_;
        if (cbest) best_cur_ = cres.bb_per_hand;
        std::printf("  [iter %4d] LBR cur=%.4f bb/hand  (win %.3f, %.0fs)%s\n",
                    iter, cres.bb_per_hand, cres.lbr_win_rate,
                    cres.wall_ms / 1000.0, cbest ? "  *best*" : "");
        std::fflush(stdout);
        if (cbest && !cfg_.ckpt_dir.empty()) {  // deployable σ: RM⁺ readout
            std::filesystem::create_directories(cfg_.ckpt_dir);
            torch::save(played, cfg_.ckpt_dir + "/cur_best.pt");
        }
    }
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
    if (cfg_.regret_ema > 0.f)  // EMA σ state, so resume keeps the smoothing
        torch::save(regret_ema_, cfg_.ckpt_dir + "/regret_ema.pt");
    if (cfg_.regret_buffer)  // the reservoir IS the cumulative-regret state
        regret_buf_->save(cfg_.ckpt_dir + "/regret_buf.pt");
    // Adam moments: in neural-cum mode the regret net IS the cumulative-
    // regret state, so a cold-Adam restart writes shocked steps straight
    // into it (observed: resume at iter 1000 cost ~1.5 bb of cur-σ LBR and
    // inflated rmag 5.3→7.8). Persist all three optimizers.
    auto save_opt = [&](const ForeachAdam& opt, const std::string& path) {
        std::vector<torch::Tensor> st;
        for (const auto& t : opt.exp_avg())    st.push_back(t.clone());
        for (const auto& t : opt.exp_avg_sq()) st.push_back(t.clone());
        st.push_back(torch::tensor(opt.step_count()));
        torch::save(st, path);
    };
    save_opt(*value_opt_,  cfg_.ckpt_dir + "/opt_value.pt");
    save_opt(*regret_opt_, cfg_.ckpt_dir + "/opt_regret.pt");
    save_opt(*avg_opt_,    cfg_.ckpt_dir + "/opt_avg.pt");
    std::ofstream(cfg_.ckpt_dir + "/iter.txt")
        << iter << "\n" << best_lbr_ << "\n" << best_cur_;
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
    { torch::NoGradGuard ng;  // resync target + EMA nets to the loaded nets
      auto s = value_->parameters(); auto d = value_target_->parameters();
      for (size_t i = 0; i < s.size(); ++i) d[i].copy_(s[i]);
      auto rs = regret_->parameters(); auto rd = regret_ema_->parameters();
      for (size_t i = 0; i < rs.size(); ++i) rd[i].copy_(rs[i]); }
    if (std::filesystem::exists(cfg_.ckpt_dir + "/regret_ema.pt")) {
        // Real EMA state (newer checkpoints); else the regret_ copy above
        // stands and the EMA re-forms over ~1/(1-β) iters.
        torch::load(regret_ema_, cfg_.ckpt_dir + "/regret_ema.pt");
        regret_ema_->to(device_);
    }
    if (cfg_.regret_buffer &&
        std::filesystem::exists(cfg_.ckpt_dir + "/regret_buf.pt"))
        regret_buf_->load(cfg_.ckpt_dir + "/regret_buf.pt");
    auto load_opt = [&](ForeachAdam& opt, const std::string& path) {
        if (!std::filesystem::exists(path)) return;   // pre-fix checkpoint
        std::vector<torch::Tensor> st;
        torch::load(st, path);
        const size_t n = opt.exp_avg().size();
        if (st.size() != 2 * n + 1) return;           // param set changed
        std::vector<torch::Tensor> m(st.begin(), st.begin() + n);
        std::vector<torch::Tensor> v(st.begin() + n, st.begin() + 2 * n);
        opt.load_state(m, v, st.back().item<int64_t>());
    };
    load_opt(*value_opt_,  cfg_.ckpt_dir + "/opt_value.pt");
    load_opt(*regret_opt_, cfg_.ckpt_dir + "/opt_regret.pt");
    load_opt(*avg_opt_,    cfg_.ckpt_dir + "/opt_avg.pt");
    std::ifstream itf(cfg_.ckpt_dir + "/iter.txt");
    itf >> resume_iter_;
    double b;                                          // pre-fix files lack these
    if (itf >> b) best_lbr_ = b;
    if (itf >> b) best_cur_ = b;
    std::printf("  [resume] loaded checkpoint at iter %d (best avg %.3f, "
                "cur %.3f)\n", resume_iter_, best_lbr_, best_cur_);
    return true;
}

// ── outer ESCHER loop ───────────────────────────────────────────────────────
void EscherTrainer::train() {
    std::printf("ESCHER HUNL: %d iters, envs=%d, value/regret/avg traj=%d/%d/%d\n",
                cfg_.iterations, cfg_.rollout_envs, cfg_.value_traj, cfg_.regret_traj,
                cfg_.avg_traj);
    std::printf("  value: lam=%.2f tau=%.3f eps=%.3f buf_cap=%d | regret: %s gamma=%.3f ema=%.3f pred=%.2f\n",
                cfg_.value_lambda, cfg_.value_tau, cfg_.value_eps, cfg_.value_buf_cap,
                cfg_.regret_buffer ? "reinit-buffer(RM)" : "neural-cum(RM+)",
                cfg_.ncum_gamma, cfg_.regret_ema, cfg_.predictive);
    try_resume();
    if (resume_iter_ > 0) {
        // Prefill the non-checkpointed reservoirs to steady state BEFORE any
        // fit touches them. Resuming with a thin refilling value buffer feeds
        // ~7x-noisier Q into the gamma=1 regret cumulative, which rectifies
        // and KEEPS it (observed at resume@2500: rloss 0.16->0.52 and rmag
        // 6.9->9.1 within ~50 iters, cur-LBR 1.5->5.3, despite warm
        // Adam/EMA). Rollout-only passes are ~0.2s each; fresh on-sigma data
        // beats checkpointing stale buffers.
        iter_ = resume_iter_;   // recency weights for the prefill rows
        auto t0 = std::chrono::steady_clock::now();
        if (value_buf_)
            while (value_buf_->size() < cfg_.value_buf_cap)
                collect_and_train_value(/*fit=*/false);
        if (resume_iter_ > cfg_.avg_warmup)
            while (avg_buf_->size() < cfg_.buf_cap)
                collect_avg(resume_iter_);
        std::chrono::duration<double> el =
            std::chrono::steady_clock::now() - t0;
        std::printf("  [resume] reservoirs prefilled (value %d, avg %d) "
                    "in %.1fs\n",
                    value_buf_ ? value_buf_->size() : 0, avg_buf_->size(),
                    el.count());
        std::fflush(stdout);
    }
    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto el = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count(); };
    for (iter_ = resume_iter_ + 1; iter_ <= cfg_.iterations; ++iter_) {
        auto t0 = clk(); collect_and_train_value();
        auto t1 = clk(); collect_regret(0); collect_regret(1);
        auto t2 = clk(); fit_regret();
        if (cfg_.regret_ema > 0.f) {  // regret_ema_ ← β·regret_ema_ + (1−β)·regret_
            torch::NoGradGuard ng;
            const float b = cfg_.regret_ema;
            auto s = regret_->parameters(); auto d = regret_ema_->parameters();
            for (size_t i = 0; i < s.size(); ++i) d[i].mul_(b).add_(s[i], 1.f - b);
            auto sb = regret_->buffers(); auto db = regret_ema_->buffers();
            for (size_t i = 0; i < sb.size(); ++i) db[i].copy_(sb[i]);
        }
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
