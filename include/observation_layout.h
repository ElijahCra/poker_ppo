#pragma once
//
// observation_layout.h — single source of truth for the obs-vector layout.
//
// The PPO observation tensor is a flat [obs_dim] vector with three to five
// sub-blocks laid out back-to-back:
//
//   ┌──────────┬────────────┬────────────┬───────────────┬────────────┐
//   │ hole(52) │ community  │ static     │ round_summary │ history    │
//   │          │ cards(52)  │ features   │ block?        │ block?     │
//   └──────────┴────────────┴────────────┴───────────────┴────────────┘
//   0          52           104          ...             ...          obs_dim
//
// Two sides need to agree on these offsets:
//   - `PokerEnvironment` writes them in `compute_observation` /
//     `write_round_summary_block` / `write_history_block`.
//   - `TowerImpl::forward` reads them via `obs.narrow(...)` to split the
//     obs back into the parts each network sub-module consumes.
//
// Pre-refactor each side computed its own offsets locally — the network
// even derived `static_dim` by *subtraction* (`obs_dim - rs - hist`), so a
// new sub-block silently shifted the read positions on the network side.
// `ObservationLayout` collapses both sides onto the same constexpr struct;
// the env asks for write offsets and the network asks for read offsets.
//

#include "config.h"   // BetHistoryConfig, RoundSummaryConfig
#include "features.h"

namespace poker_ppo {

struct ObservationLayout {
    // Fixed structural constants — independent of the feature flags.
    static constexpr int CARD_SLOTS              = 52;  // 1 slot per card id
    // stacks(2) + pot + cur_bet + raises + round one-hot(4) + seat = 10
    static constexpr int FEAT_BASIC              = 10;
    // Per-round normalised hand_indexer bucket id; 4 rounds = 4 floats.
    static constexpr int FEAT_HAND_STRENGTH      =
        features::HAND_STRENGTH ? 4 : 0;
    static constexpr int FEAT_STATIC             =
        FEAT_BASIC + FEAT_HAND_STRENGTH;

    // ── Offsets (in floats) into the obs vector ─────────────────────────
    int hole_off;             // current player's hole-card one-hot block
    int community_off;        // community-card one-hot block
    int static_off;           // basic numeric features + hand-strength
    int round_summary_off;    // [optional] round-summary block start
    int history_off;          // [optional] attention-history block start

    // ── Sub-block sizes (already include feature-flag gating) ──────────
    int round_summary_dim;    // 0 if RoundSummaryConfig disabled
    int history_dim;          // 0 if BetHistoryConfig disabled

    // Total length of the obs tensor — the network's input dim.
    int total_dim;

    // Build the layout from the runtime configs. constexpr-friendly so the
    // result can be `inline constexpr` when the configs are.
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

}  // namespace poker_ppo
