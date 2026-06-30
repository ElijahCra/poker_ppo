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
    float    lr            = 1e-3f;
    // neural RM⁺ cumulative regret discount (1.0 = pure RM⁺/CFR+; <1 = DCFR)
    float    ncum_gamma    = 1.0f;
    // value target = λ·MC + (1−λ)·bootstrap (TD-λ). λ=1 → pure-ESCHER MC
    // (default); λ=0 → DREAM bootstrap Q(h,a)→V̄(child) (collapses alone on
    // HUNL — deadly triad); 0<λ<1 → grounded bootstrap (MC anchors, bootstrap
    // cuts variance + covers rare actions).
    float    value_lambda  = 1.0f;
    int      buf_cap       = 4'000'000;
    int      eval_every    = 50;     // iterations between LBR evaluations
    int      lbr_hands     = 10000;
    uint64_t seed          = 0;
    std::string ckpt_dir;            // empty = no checkpointing
    int      ckpt_every    = 100;
};

class EscherTrainer {
public:
    EscherTrainer(IPokerEnvironmentFactory& factory, EscherConfig cfg,
                  torch::Device device);

    void train();

    // Average policy (the deployed strategy LBR attacks).
    ActorCritic& average_network() { return avg_; }

private:
    // ── current strategy σ = RM⁺ on the regret net's actor logits ──
    // Returns a CPU probability row [A] over the legal actions (0 elsewhere).
    std::vector<float> sigma(ActorCritic& regret_net,
                             const torch::Tensor& obs_row,
                             const torch::Tensor& mask_row);
    int sample(const std::vector<float>& probs, std::mt19937& rng);
    int sample_uniform(const torch::Tensor& mask_row, std::mt19937& rng);

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

    // Reservoir buffers (feature rows kept on CPU; uploaded per-minibatch).
    struct Sample {
        torch::Tensor feat;   // [obs_dim]   (full obs; heads slice as needed)
        torch::Tensor target; // [A] regret  OR scalar (value)  OR unused
        int64_t       action; // value/avg: taken action; regret: -1
        float         z;      // value: MC return; else 0
        float         weight; // iteration weight (linear)
    };
    struct Reservoir {
        explicit Reservoir(size_t cap, uint64_t seed) : cap_(cap), rng_(seed) {}
        void add(Sample s);
        std::vector<Sample> data;
        size_t cap_; long n_ = 0; std::mt19937_64 rng_;
    };

    IPokerEnvironmentFactory&          factory_;
    EscherConfig                       cfg_;
    torch::Device                      device_;
    BetConfig                          bet_cfg_;
    int                                A_ = 0;
    int                                obs_dim_ = 0;
    std::unique_ptr<IPokerEnvironment> env_;
    std::mt19937                       rng_;

    ActorCritic              value_{nullptr};
    ActorCritic              regret_{nullptr};
    ActorCritic              avg_{nullptr};
    std::unique_ptr<torch::optim::Adam> value_opt_, regret_opt_, avg_opt_;

    // value_/regret_ buffers are per-iteration (cleared); avg accumulates.
    std::vector<Sample>        value_smp_, regret_smp_;
    std::unique_ptr<Reservoir> avg_buf_;
    long iter_ = 0;
    float last_val_loss_ = 0.f, last_reg_loss_ = 0.f, last_reg_mag_ = 0.f;
};

}  // namespace poker_ppo
