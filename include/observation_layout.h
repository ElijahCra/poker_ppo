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
// The env asks for write offsets and the network asks for read offsets;
// they cannot drift.
//

#include "config.h"   // BetHistoryConfig, RoundSummaryConfig
#include "features.h"

namespace poker_ppo {

struct ObservationLayout {
    // Fixed structural constants — independent of the feature flags.
    static constexpr int CARD_SLOTS              = 52;  // 1 slot per card id
    // stacks(2) + pot + cur_bet + raises + round one-hot(4) + seat = 10
    static constexpr int FEAT_BASIC              = 10;
    // Per-round hand-info block:
    //   0  made-hand category / 9       (monotonic in made-hand strength)
    //   1  flush_draw                    (1 if 4-of-suit visible & not yet made)
    //   2  straight_draw_outs / 8        (rank-outs; OESD vs gutshot — postflop)
    //   3  straight_alive_windows / 4    (long-term straight progress; the only
    //                                      straight feature meaningful preflop —
    //                                      distinguishes 67 from 6J)
    //   4  overcards / 2                 (hole cards > max board rank)
    // 4 rounds × 5 features = 20 floats. Stripped from the binary when
    // `features::HAND_STRENGTH` is false.
    static constexpr int HAND_FEATS_PER_ROUND    = 5;
    static constexpr int HAND_FEAT_ROUNDS        = 4;
    static constexpr int FEAT_HAND_STRENGTH      =
        features::HAND_STRENGTH
            ? HAND_FEATS_PER_ROUND * HAND_FEAT_ROUNDS
            : 0;
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
