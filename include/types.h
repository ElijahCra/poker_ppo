#pragma once
//
// types.h — env↔PPO interaction types only (BetConfig, Action,
// StepResult). PPO hyperparameters and feature configs live in config.h.
//

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <vector>

namespace poker_ppo {

/// Controls the discrete action space for betting.
///
/// Non-raise actions are always: {Fold, Check/Call}.
/// Raise actions use geometrically-spaced bet sizes:
///   bet_i = min_raise * ratio^i   for i in [0, num_raise_sizes)
///
struct BetConfig {
    int    num_raise_sizes    = 4;     // distinct raise amounts
    double min_raise          = 1.0;   // smallest raise (in your unit)
    double geometric_ratio    = 2.0;   // multiplier between successive raises
    int    max_bets_per_round = 4;     // cap on raises per player per round

    /// Total discrete actions = fold + check/call + raises.
    constexpr int action_count() const noexcept { return 2 + num_raise_sizes; }

    /// Raise amount for raise index i ∈ [0, num_raise_sizes). Not constexpr
    /// because std::pow isn't constexpr until C++26.
    double raise_amount(int i) const {
        assert(i >= 0 && i < num_raise_sizes);
        return min_raise * std::pow(geometric_ratio, i);
    }

    std::vector<double> all_raise_amounts() const {
        std::vector<double> v(num_raise_sizes);
        for (int i = 0; i < num_raise_sizes; ++i) v[i] = raise_amount(i);
        return v;
    }
};

/// Action indices:
///   0         → Fold
///   1         → Check / Call
///   2 .. 2+N  → Raise(raise_amount(i - 2))
///
namespace Action {
    constexpr int Fold      = 0;
    constexpr int CheckCall = 1;

    inline int  Raise(int raise_idx)        { return 2 + raise_idx; }
    inline bool is_raise(int action_id)     { return action_id >= 2; }
    inline int  raise_index(int action_id)  {
        assert(is_raise(action_id));
        return action_id - 2;
    }
}

/// Returned by `IPokerEnvironment::reset()` and `step()`.
struct StepResult {
    torch::Tensor observation;        // [obs_dim]    float
    float         reward  = 0.0f;     // for the acting player
    bool          done    = false;
    torch::Tensor legal_action_mask;  // [action_count]  1=legal, 0=illegal
};

} // namespace poker_ppo
