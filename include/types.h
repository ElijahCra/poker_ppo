#pragma once
// Env↔PPO interaction types. Hyperparameters live in config.h.

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <vector>

namespace poker_ppo {

// Discrete action space: {Fold, Check/Call, Raise_0, ..., Raise_{N-1}}.
// Raise sizes are geometric: bet_i = min_raise * ratio^i.
struct BetConfig {
    int    num_raise_sizes    = 4;
    double min_raise          = 1.0;
    double geometric_ratio    = 2.0;
    int    max_bets_per_round = 4;

    constexpr int action_count() const noexcept { return 2 + num_raise_sizes; }

    // Not constexpr — std::pow isn't until C++26.
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

struct StepResult {
    torch::Tensor observation;
    float         reward  = 0.0f;     // for the acting player
    bool          done    = false;
    torch::Tensor legal_action_mask;  // [action_count]  1=legal, 0=illegal
};

} // namespace poker_ppo
