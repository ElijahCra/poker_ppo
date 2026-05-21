#pragma once
// Obs-tensor layout + builder. Layout:
//
//   [hole(52) | community(52) | static | round_summary? | history?]
//
// Both env (write) and network (read) go through ObservationLayout, so
// offsets can't drift between the two sides.

#include "config.h"
#include <torch/torch.h>
#include <vector>

namespace Game {
class GameContext;
}

namespace poker_ppo {

struct ObservationLayout {
    static constexpr int CARD_SLOTS              = 52;

    // stacks(2) + pot + cur_bet + raises + round one-hot(4) + seat = 10
    static constexpr int FEAT_BASIC              = 10;

    // Per-round hand features: cat/9, flush_draw, straight_outs/8,
    // straight_alive_windows/4, overcards/2. 4 rounds × 5 = 20.
    static constexpr int HAND_FEATS_PER_ROUND    = 5;
    static constexpr int HAND_FEAT_ROUNDS        = 4;
    static constexpr int FEAT_HAND_STRENGTH      =
        features::HAND_STRENGTH
            ? HAND_FEATS_PER_ROUND * HAND_FEAT_ROUNDS
            : 0;
    static constexpr int FEAT_STATIC             =
        FEAT_BASIC + FEAT_HAND_STRENGTH;

    int hole_off;
    int community_off;
    int static_off;
    int round_summary_off;
    int history_off;

    int round_summary_dim;     // 0 if disabled
    int history_dim;           // 0 if disabled

    int total_dim;

    [[nodiscard]] static constexpr ObservationLayout
    build(BetHistoryConfig hist, RoundSummaryConfig rs) noexcept
    {
        ObservationLayout L{};
        L.hole_off          = 0;
        L.community_off     = CARD_SLOTS;
        L.static_off        = 2 * CARD_SLOTS;
        L.round_summary_off = L.static_off + FEAT_STATIC;
        L.round_summary_dim = rs.dim();
        L.history_off       = L.round_summary_off + L.round_summary_dim;
        L.history_dim       = hist.history_block_dim();
        L.total_dim         = L.history_off + L.history_dim;
        return L;
    }
};

// One betting action recorded by step(). Aggregated into the round-
// summary block and serialised into the attention block.
struct BetHistoryEntry {
    uint32_t amount;        // chip delta added to pot (mbb)
    uint8_t  player;
    uint8_t  round;         // 0..3
    bool     is_aggressive; // Raise vs Call/Check/Fold
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

    // current_bet is on ActionState (state-machine-private), so the env
    // passes it in rather than us reading from ctx.
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

    BetHistoryConfig    hist_cfg_;
    RoundSummaryConfig  rs_cfg_;
    float               stack_norm_;
    float               pot_norm_;
    int                 max_raises_norm_;
    ObservationLayout   layout_;
};

} // namespace poker_ppo
