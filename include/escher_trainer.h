#pragma once
//
// ESCHER trainer (McAleer et al. 2022, arXiv:2206.04122) — pure paper ESCHER
// wired to the HUNL env, validated on Kuhn/Leduc first (research/escher).
//
// Three networks (each a reused ActorCritic for the conv encoder + heads):
//   value_  : the PRIVILEGED Q-critic head Q_θ(h,·) = history value function,
//             trained with MONTE-CARLO terminal-utility targets under the
//             current strategy (no bootstrapping — ESCHER's point vs DREAM).
//   regret_ : the actor head as regret R(s,·); σ = RM⁺ on it. ONE shared net
//             over both seats (obs encodes seat), as in the existing self-play.
//             Cumulative regret is held in the net (neural RM⁺), so the regret
//             net is NOT re-init each iter (this is the RM⁺ upgrade over the
//             paper's plain-RM reinit-buffer — a standard CFR+ improvement; the
//             Leduc study showed plain RM floors at vanilla-CFR's slow rate).
//             Q(s,·) is deterministic given obs, so the sampled instantaneous
//             regret is EXACT per infoset — only visitation is sampled.
//   avg_    : the actor head as the average policy π̄, trained by classification
//             on (infoset, a_taken). THIS is the deployed, LBR-judged strategy.
//
// Sampling (the ESCHER fixed sampler): value rollouts under the current σ;
// regret rollouts under a FIXED sampler — traverser ~ uniform, opponent ~ σ.
// The privileged critic gives Q(s,·) for ALL actions at the visited node, so
// no tree branching is needed (outcome-sampled visitation, value-fn action
// values, no importance weight).
//
#include "config.h"
#include "environment.h"
#include "network.h"
#include "optim.h"

#include <torch/torch.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace poker_ppo {

struct EscherConfig {
    int      iterations    = 2000;
    // trajectories per iteration for each phase (env-stepped, alloc-light)
    int      value_traj    = 6000;
    int      regret_traj   = 6000;   // per player
    int      avg_traj      = 6000;
    // SGD steps + minibatch for each net per iteration
    int      value_steps   = 1500;
    int      regret_steps  = 1500;
    int      avg_steps      = 1500;
    int      batch_size    = 2048;
    // Parallel rollout envs. The rollout alternates a batched inference with
    // CPU env stepping; more envs = fewer, fatter inference batches (the
    // forwards at 512 rows are launch-bound on a 3060 Ti). Waste tail: envs
    // mid-hand when n_traj is reached are discarded, so keep envs ≪ traj.
    int      rollout_envs  = 512;
    float    lr            = 1e-3f;
    // neural RM⁺ cumulative regret discount (1.0 = pure RM⁺/CFR+; <1 = DCFR)
    float    ncum_gamma    = 1.0f;
    // regret update: false = neural-cumulative (warm net holds RM⁺ cumulative,
    // my variant); true = the PAPER's Deep-CFR reservoir buffer (accumulate raw
    // instantaneous regrets, REINIT the regret net each iter, RM at read).
    bool     regret_buffer = false;
    // Regret-net EMA — σ-smoothing stabiliser for the oscillating σ sequence
    // (the confirmed HUNL floor). Maintain regret_ema_ = β·regret_ema_ +
    // (1−β)·regret_ and PLAY + AVERAGE the smoothed σ=RM⁺(regret_ema_), while the
    // neural-cum cumulative still accumulates in the TRUE regret_ (using the EMA
    // as the cumulative would act like an extra discount → over-soften). 0 = off;
    // β∈(0,1), higher = smoother/laggier (β=0.9 ≈ 10-iter smoothing).
    float    regret_ema    = 0.0f;
    // Predictive/Optimistic RM⁺ (Farina et al. 2021, "Faster Game Solving via
    // Predictive Regret Matching") — the theoretically-correct fix for the CFR
    // oscillation. Play RM⁺(cum + η·m) where the predictor m = the CURRENT
    // instantaneous regret Q−V̄ (an optimistic one-step lookahead; m is exactly
    // the increment that will be added to cum, so η is scale-free). 0 = vanilla
    // RM⁺; η=1 = standard one-step optimism. Alternative to regret_ema.
    float    predictive    = 0.0f;
    // value target = λ·MC + (1−λ)·bootstrap (TD-λ). λ=1 → pure-ESCHER MC
    // (default); λ=0 → DREAM bootstrap Q(h,a)→V̄(child) (collapses alone on
    // HUNL — deadly triad); 0<λ<1 → grounded bootstrap (MC anchors, bootstrap
    // cuts variance + covers rare actions).
    float    value_lambda  = 1.0f;
    // Polyak target-net rate for the bootstrap value targets (deadly-triad
    // stabiliser): target ← τ·value + (1−τ)·target. 0 disables (use live net).
    float    value_tau     = 0.0f;
    // ε-uniform exploration mixed into the value-collection policy so the
    // privileged Q-critic is supervised on low-σ actions too: the regret
    // r(a)=Q(a)−V̄ needs accurate Q for the very actions σ rarely plays.
    // Bias is O(ε) on the continuation; coverage of a rare action goes from
    // ~σ(a) to ≥ε/|legal|. 0 = pure on-σ (default).
    float    value_eps     = 0.0f;
    // Value-target reservoir across iterations (recency-weighted by iter, so
    // near-Nash late σ dominates and staleness bias stays small). 0 = refit
    // each iter on that iter's rollouts ONLY (high estimator variance); >0 =
    // accumulate value samples up to this cap → the critic fits ~window× more
    // data → lower-variance Q → lower-noise regret. Not checkpointed (refills
    // within a few iters). Set to ~3–5 iters worth of nodes to bound staleness.
    int      value_buf_cap = 0;
    int      buf_cap       = 1'000'000;
    int      avg_warmup    = 0;      // skip averaging the first N iters' σ (the
                                     // average is dragged by early off-Nash σ)
    // Average-policy iteration weight = t^k. k=1 = linear CFR (right for an
    // OSCILLATING σ). Under regret_ema the σ sequence is smooth and improving,
    // so the time-average mostly adds stale history — k=2..3 concentrates the
    // average on late σ (confirmed HUNL: cur-σ 2.2 stable vs linear-avg 5.5+).
    float    avg_pow       = 1.0f;
    int      eval_every    = 50;     // iterations between LBR evaluations
    int      lbr_hands     = 10000;
    // LBR hand shards run in parallel threads. Hands are independent, so
    // sharding is the SAME estimator (per-shard RNG streams) — LBR is
    // latency-bound (batch-1 net queries + CPU equity), not GPU-bound, so
    // this scales ~linearly with cores. Shard 0 prints the attribution table.
    int      lbr_threads   = 1;
    // Also LBR the CURRENT σ (RM⁺ on the played regret net) each eval — the
    // observable that separates σ-DRIFT (cur-σ degrades with the avg) from an
    // approximation FLOOR (cur-σ flat while avg converges to it). Costs one
    // extra LBR pass per eval.
    bool     lbr_cur       = false;
    uint64_t seed          = 0;
    std::string ckpt_dir;            // empty = no checkpointing
    int      ckpt_every    = 100;
};

class EscherTrainer {
public:
    EscherTrainer(IPokerEnvironmentFactory& factory, EscherConfig cfg,
                  torch::Device device);
    ~EscherTrainer();   // out-of-line: FitGraphs is incomplete here

    void train();

    // Average policy (the deployed strategy LBR attacks).
    ActorCritic& average_network() { return avg_; }

private:
    // ── current strategy σ = RM⁺ on the regret net's actor logits ──
    // Raw CPU row pointers ([A] each) — the rollout loop hands out rows of
    // contiguous batch tensors, so no per-row tensor views are built.
    std::vector<float> sigma(const float* lg, const float* mk);
    // Predictive/Optimistic RM⁺: σ = RM⁺(cum + η·(Q − V̄_base)), V̄_base under
    // σ_base=RM⁺(cum). Needs the value net's Q row → passed from the rollout.
    std::vector<float> sigma_pred(const float* lg, const float* q,
                                  const float* mk, float eta);
    int sample(const std::vector<float>& probs, std::mt19937& rng);
    int sample_uniform(const float* mk, std::mt19937& rng);

    // ── the three ESCHER phases (one outer iteration) ──
    void collect_and_train_value();      // MC, current σ, both seats (cleared)
    void collect_regret(int traverser);  // fixed sampler → regret samples
    void fit_regret();                   // neural RM⁺ cumulative update (in-net)
    void collect_avg(long t);            // current σ → avg reservoir
    void fit_avg();                      // classification on (infoset, a_taken)

    void run_lbr(int iter);
    void save_checkpoint(int iter);
    bool try_resume();   // loads the 3 nets + iter from cfg_.ckpt_dir if present
    int  resume_iter_ = 0;

    // TENSOR-BACKED reservoir: features/targets/actions/weights are contiguous
    // CPU tensors (capacity-preallocated), so minibatches are one fast
    // index_select instead of a per-row tensor build — the difference between
    // ~24s/iter and paper-scale step counts. Checkpointable (the buffer IS the
    // cumulative-regret state in paper mode). Per-iteration buffers reuse the
    // same class with clear(); past-capacity adds reservoir-subsample, which
    // is statistically benign for every consumer (all fits sample uniformly).
    class Reservoir {
    public:
        Reservoir(size_t cap, int D, int A, uint64_t seed);
        // feat:[D] target:[A] (target may be null) — raw CPU pointers (memcpy).
        void add(const float* feat, const float* target, int64_t action, float w);
        void  clear() { filled_ = 0; n_ = 0; }
        int   size() const { return static_cast<int>(filled_); }
        float target_rms() const;   // global RMS of the regret targets (RM scale)
        struct MB { torch::Tensor feat, target, action, weight; };  // on device
        MB   sample(int B, std::mt19937& rng, torch::Device dev);
        // One bulk H2D of the filled rows — the fit loops then minibatch with
        // on-device index_select (no per-step CPU gather / sync copy).
        MB   upload(torch::Device dev) const;
        void save(const std::string& path) const;
        void load(const std::string& path);
    private:
        size_t cap_, filled_ = 0;
        int    D_, A_;
        long   n_ = 0;
        std::mt19937_64 rng_;
        torch::Tensor feats_, targets_, actions_, weights_;
    };

    // ── unified fit engine ──
    // One SGD pass of `steps` minibatches from `buf` into `net` under `opt`.
    // CUDA: bulk-upload the buffer once, sample minibatches on-device, and
    // replay a captured fwd+bwd graph per step (ForeachAdam consumes the
    // static grad tensors). CPU device: falls back to Reservoir::sample.
    enum class FitKind { Value, Regret, Avg };
    float run_fit(FitKind kind, ActorCritic& net, ForeachAdam& opt,
                  Reservoir& buf, int steps, bool scale_by_rms,
                  float* out_target_mag = nullptr);
    // Paper-mode per-iter reinit that keeps parameter ADDRESSES stable (the
    // captured fit graph reads params by pointer): draw a fresh net, copy_
    // its params/buffers into regret_, rebuild the optimizer state.
    void reinit_regret_inplace();
    struct FitGraphs;
    std::unique_ptr<FitGraphs> graphs_;

    IPokerEnvironmentFactory&          factory_;
    EscherConfig                       cfg_;
    torch::Device                      device_;
    BetConfig                          bet_cfg_;
    int                                A_ = 0;
    int                                obs_dim_ = 0;
    std::unique_ptr<IPokerEnvironment> env_;
    std::mt19937                       rng_;

    ActorCritic              value_{nullptr};
    ActorCritic              value_target_{nullptr};   // Polyak snapshot (τ>0)
    ActorCritic              regret_{nullptr};
    ActorCritic              regret_ema_{nullptr};   // σ-smoothing shadow of regret_
    ActorCritic              avg_{nullptr};
    // ForeachAdam (multi-tensor, fused) not stock torch::optim::Adam: the nets
    // are small, so stock Adam's per-parameter kernel loop is pure launch
    // overhead and the SGD fits dominate each iter. See include/optim.h.
    std::unique_ptr<ForeachAdam> value_opt_, regret_opt_, avg_opt_;

    // avg + paper-mode regret reservoirs accumulate across iterations;
    // *_iter_buf_ are per-iteration staging (cleared each iter) for the value
    // MC targets and the neural-cum regret targets.
    std::unique_ptr<Reservoir> avg_buf_, regret_buf_, value_buf_;
    std::unique_ptr<Reservoir> value_iter_buf_, regret_iter_buf_;
    long iter_ = 0;
    float last_val_loss_ = 0.f, last_reg_loss_ = 0.f, last_reg_mag_ = 0.f;
    double best_lbr_ = 1e9;   // best avg-net LBR so far → snapshot avg_best.pt
    double best_cur_ = 1e9;   // best cur-σ LBR so far → snapshot cur_best.pt
                              // (the played EMA σ via RM⁺ readout — the
                              // deployable artifact; late runs creep, so the
                              // trough state must be captured, not the last)
};

}  // namespace poker_ppo
