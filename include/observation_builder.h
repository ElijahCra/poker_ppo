#pragma once
//
// observation_builder.h — owns the obs-tensor construction logic that
// PokerEnvironment used to inline. One builder per env; one obs tensor
// returned per `build()` call.
//
// The builder reads from a Game::GameContext + a hand-local bet-history
// vector + the env's normaliser constants; it writes through an
// `ObservationLayout` shared with the network so the two sides cannot
// drift on offsets.
//

#include "config.h"
#include "observation_layout.h"

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Game {
class GameContext;   // fwd-decl; full type in Game/Context/GameContext.hpp
}

namespace poker_ppo {

// One past betting action recorded by PokerEnvironment::step. The builder
// serialises a window of these into the attention block (when enabled)
// and aggregates them into per-round summary stats (when enabled).
struct BetHistoryEntry {
    uint32_t amount;        // chip delta added to pot (mbb)
    uint8_t  player;        // 0 or 1
    uint8_t  round;         // 0..3
    bool     is_aggressive; // true = Raise, false = Call/Check/Fold
};

class ObservationBuilder {
public:
    ObservationBuilder(BetHistoryConfig    hist_cfg,
                       RoundSummaryConfig  rs_cfg,
                       float               stack_norm,
                       float               pot_norm,
                       int                 max_raises_norm);

    [[nodiscard]] const ObservationLayout& layout()  const noexcept { return layout_; }
    [[nodiscard]] int                       obs_dim() const noexcept { return layout_.total_dim; }

    // Allocate + fill a [obs_dim] float tensor for the given context.
    // `current_bet` is the live ActionState's currentBet — it isn't on
    // the GameContext directly (state-machine-private), so the env passes
    // it in.
    [[nodiscard]] torch::Tensor build(
        const Game::GameContext&             ctx,
        const std::vector<BetHistoryEntry>&  bet_history,
        int                                  current_bet) const;

private:
    void write_round_summary(torch::TensorAccessor<float, 1>& a,
                             const std::vector<BetHistoryEntry>& bet_history,
                             int current_player) const;
    void write_history(torch::TensorAccessor<float, 1>& a,
                       const std::vector<BetHistoryEntry>& bet_history,
                       int current_player) const;

    // Lazily populate `indexer_norm_` from the static hand_indexer's
    // per-round canonical-bucket counts. The indexer is initialised on
    // first DiscreteGame construction, so we wait until first build()
    // call to read it.
    void ensure_indexer_norm() const;

    BetHistoryConfig    hist_cfg_;
    RoundSummaryConfig  rs_cfg_;
    float               stack_norm_;
    float               pot_norm_;
    int                 max_raises_norm_;
    ObservationLayout   layout_;

    // 1 / per-round hand_indexer bucket count, populated lazily on first
    // build() call. Mutable because it's a pure memoisation cache — no
    // observable state change for callers.
    mutable std::array<float, 4> indexer_norm_{};
};

} // namespace poker_ppo
