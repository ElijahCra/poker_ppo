// ReBeL play-time agent as an LBR target (Phase B′).
//
// Act-by-re-solving: at every query the agent solves the street subgame
// rooted at the env's CURRENT node with its tracked PBS and reads the
// average policy — for its actual hand (act) or for all 1326 combos at
// once (probs_for_holes: LBR's Bayes update and raise-response pricing
// come from ONE solve, cheaper than the per-combo net path). River
// subgames solve exactly; turn subgames use the value net at StreetEnd
// leaves. Solves are cached per street keyed by the env's action_log —
// snapshot-consistent, so LBR's hypothetical push_state probes name (and
// cache) their own nodes without touching the real line.
//
// PBS tracking (the ReBeL self-play model): both ranges start uniform,
// are masked as board cards appear, and multiply by the solve's own
// average policy at each REAL action (note_action, both seats). The
// opponent is modeled as playing the same equilibrium; LBR of course
// doesn't — exploiting that model is exactly what the bound measures.
// Off-abstraction opponent actions (LBR raise sizes outside the sparse
// set) leave the range unchanged and re-root the next solve at the
// post-action state — the standard continual-resolving fallback.
//
// Hybrid deployment (until flop/preflop nets exist): streets 0-1 play and
// update ranges under a BLUEPRINT policy (the PPO ActorCritic through the
// LBR adapter). Without one, a check/call stub keeps the pipeline
// runnable — the resulting bound measures that toy hybrid, not ReBeL.
#pragma once

#include "hunl_solver.h"
#include "hunl_value.h"
#include "lbr.h"
#include "poker_env.h"

#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace rebel_hunl {

struct RebelPlayConfig {
    int t_turn        = 120;   // CFR iterations, turn solves (net leaves)
    int t_river       = 200;   // river solves (exact)
    int t_flop        = 60;    // flop solves (net turn-root leaves)
    int refresh_every = 5;     // CFR-AVG leaf refresh cadence (turn)
    int refresh_flop  = 10;    // flop refreshes cost 49 turn-root queries
    // Turn keeps the training abstraction (the net's leaf values assume
    // it). River solves are exact to terminal — no net consistency to
    // preserve — so the river set adds 0.5-pot (action 4): LBR's analytic
    // raise menu is {~0.5-pot, ~pot, all-in}, and an off-tree raise both
    // skips the belief update and leaves the response unpriced (measured:
    // ALL residual LBR profit was river-ending hands).
    std::vector<int> actions       = {0, 1, 7, 13};
    std::vector<int> actions_river = {0, 1, 4, 7, 13};
    // Flop trees exclude all-in: a called shove 2 cards early is a
    // multi-street AllinShowdown the solver deliberately guards (later
    // stage); LBR never raises pre-river in analytic mode, so no real
    // line reaches one. Pot-raise chains that would exceed the stack are
    // masked illegal by the env, so none sneak back in.
    std::vector<int> actions_flop  = {0, 1, 7};
    // Safe re-solving (Burch et al.) on river solves: the opponent's
    // alternatives come from the turn solve's net-priced leaf (street
    // entry) or the previous river solve's values_at (deeper re-solves);
    // turn solves stay model-based until flop-entry alternatives exist.
    bool   gadget     = true;
    double gadget_mix = 0.1;
    // false → blueprint plays the flop too (re-solving starts at the
    // turn). Measured 2026-07-09: flop solves at T=150 on the current
    // street-2 net ERASED the high-T gain (1.424 → 1.953; river share
    // 1.60 → 3.41) — the Leduc oracle-sharpening effect: iterations
    // amplify a thin net's bias. Re-enable as street-2 data grows.
    bool   flop_solve = true;
    // true → re-solve from the PREFLOP root too (no blueprint anywhere).
    // Preflop StreetEnd chances are 3-card flops: the solver values
    // pf_samples sampled flops via the net's value_boards (see
    // HunlSolver::pf_samples). Uses the flop action set — same all-in /
    // multi-street-showdown reasoning, one street earlier.
    bool   preflop_solve = false;
    int    t_preflop     = 40;
    int    pf_samples    = 64;   // sampled flops per preflop leaf
    uint64_t seed     = 0;
};

// Blueprint stub: check/call everything. Pipeline smoke only.
class CheckCallTarget final : public poker_ppo::ILBRTarget {
public:
    torch::Tensor probs_for_holes(
        poker_ppo::PokerEnvironment& env, const torch::Tensor& mask,
        const std::vector<std::array<uint8_t, 2>>& holes) override;
    int act(poker_ppo::PokerEnvironment& env,
            const torch::Tensor& mask) override;
};

class RebelTarget final : public poker_ppo::ILBRTarget {
public:
    // net: the (hand-strength) PBS value net for turn leaves. blueprint:
    // policy for streets 0-1 (non-owning; shared across shards is fine —
    // it is queried read-only). rng seed should differ per shard.
    RebelTarget(HunlValueNet net, double stack, torch::Device device,
                RebelPlayConfig cfg, poker_ppo::ILBRTarget* blueprint);

    torch::Tensor probs_for_holes(
        poker_ppo::PokerEnvironment& env, const torch::Tensor& mask,
        const std::vector<std::array<uint8_t, 2>>& holes) override;
    int act(poker_ppo::PokerEnvironment& env,
            const torch::Tensor& mask) override;
    void on_hand_start(poker_ppo::PokerEnvironment& env) override;
    void note_action(poker_ppo::PokerEnvironment& env, int action) override;

private:
    struct Solve {
        std::vector<int> log_at_root;   // env.action_log() at the root
        std::unique_ptr<HunlSolver> solver;
    };

    // first street the agent re-solves; earlier streets play/track under
    // the blueprint
    int solve_from() const {
        return cfg_.preflop_solve ? 0 : cfg_.flop_solve ? 1 : 2;
    }
    // mask newly revealed board cards out of both ranges; clear the solve
    // cache on street changes
    void sync_public(poker_ppo::PokerEnvironment& env);
    // opponent per-combo alternatives for the river gadget, priced by the
    // net at the turn solve's StreetEnd (CFR-AVG beliefs, dealt card).
    // Called on the 2→3 street transition, before the turn cache drops.
    void alt_from_turn_leaf(poker_ppo::PokerEnvironment& env);
    // solver + node id for the env's CURRENT node: cached solve whose
    // log_at_root prefixes the current action_log and whose tree contains
    // the suffix walk, else a fresh solve rooted here.
    std::pair<HunlSolver*, int> solve_at(poker_ppo::PokerEnvironment& env);
    // multiply `seat`'s tracked range by P(action | node) under the given
    // per-combo policy column; no-op (revert) if the update zeroes it.
    void apply_range_update(int seat, const std::vector<double>& col);

    HunlValueNet  net_;
    double        stack_;
    torch::Device device_;
    RebelPlayConfig cfg_;
    poker_ppo::ILBRTarget* blueprint_;
    std::unique_ptr<HunlNetOracle> oracle_;
    std::mt19937 rng_;

    HunlPBS pbs_;
    int     board_seen_  = 0;
    int     cache_round_ = -1;
    std::vector<Solve> cache_;
    // the target's seat this hand (set at its first policy query; -1 =
    // unknown → gadget skipped defensively) and the opponent's
    // alternative values for gadgeted river solves, both players' sides
    // kept so the seat can be selected at enable time
    int  seat_ = -1;
    bool have_alt_ = false;
    std::array<std::vector<double>, 2> alt_;
};

}  // namespace rebel_hunl
