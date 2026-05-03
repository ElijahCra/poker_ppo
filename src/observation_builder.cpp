#include "observation_builder.h"

#include "BettingConfig.hpp"          // pulls Game::MAX_BET_ACTIONS for InfoSetData
#include "Context/GameContext.hpp"
#include "Utility/HandAbstraction/hand_index.h"

#include <algorithm>

namespace poker_ppo {

ObservationBuilder::ObservationBuilder(BetHistoryConfig    hist_cfg,
                                       RoundSummaryConfig  rs_cfg,
                                       float               stack_norm,
                                       float               pot_norm,
                                       int                 max_raises_norm)
    : hist_cfg_(hist_cfg),
      rs_cfg_(rs_cfg),
      stack_norm_(stack_norm),
      pot_norm_(pot_norm),
      max_raises_norm_(std::max(1, max_raises_norm)),
      layout_(ObservationLayout::build(hist_cfg, rs_cfg))
{}

void ObservationBuilder::ensure_indexer_norm() const {
    if constexpr (!features::HAND_STRENGTH) return;
    if (indexer_norm_[0] != 0.0f) return;  // already populated

    for (int r = 0; r < 4; ++r) {
        const uint64_t sz = hand_indexer_size(
            &::Game::CardData::flopIndexer,
            static_cast<uint_fast32_t>(r));
        if (sz == 0) {
            // Indexer not yet populated — leave the cache empty so the
            // next call retries.
            indexer_norm_.fill(0.0f);
            return;
        }
        indexer_norm_[r] = 1.0f / static_cast<float>(sz);
    }
}

torch::Tensor ObservationBuilder::build(
    const Game::GameContext&             ctx,
    const std::vector<BetHistoryEntry>&  bet_history,
    int                                  current_bet) const
{
    auto obs = torch::zeros({layout_.total_dim});
    auto a   = obs.accessor<float, 1>();

    const int me  = ctx.getCurrentPlayer();
    const int opp = 1 - me;

    // Hole cards (current player only — never leak the opponent's).
    const auto hole = ctx.getHoleCards(me);
    a[layout_.hole_off + hole[0]] = 1.0f;
    a[layout_.hole_off + hole[1]] = 1.0f;

    // Community cards visible for this round.
    const int n_comm = ctx.getCommunityCount();
    for (int i = 0; i < n_comm; ++i) {
        a[layout_.community_off + ctx.getCommunityCard(i)] = 1.0f;
    }

    // Static numeric features.
    int off = layout_.static_off;
    a[off++] = static_cast<float>(ctx.getStack(me))  / stack_norm_;
    a[off++] = static_cast<float>(ctx.getStack(opp)) / stack_norm_;
    a[off++] = static_cast<float>(ctx.getPot())      / pot_norm_;
    a[off++] = static_cast<float>(current_bet)       / stack_norm_;
    a[off++] = static_cast<float>(ctx.getRaiseNum())
             / static_cast<float>(max_raises_norm_);

    const int round = ctx.getRoundNumber();
    for (int r = 0; r < 4; ++r) a[off++] = (r == round) ? 1.0f : 0.0f;

    a[off++] = static_cast<float>(me);

    // Per-round normalised hand_indexer bucket ids. The indexer's
    // suit-isomorphic clustering is hard for a small MLP to derive from
    // raw card one-hots, so this is a meaningful inductive bias for
    // free. Future-round buckets are masked to 0 so we don't leak
    // unseen community cards. Stripped from the binary when
    // `features::HAND_STRENGTH` is false.
    if constexpr (features::HAND_STRENGTH) {
        ensure_indexer_norm();
        const auto& handHashes = ctx.cards.handHashes;
        const int   round_now  = ctx.getRoundNumber();
        for (int r = 0; r < 4; ++r) {
            if (r <= round_now) {
                a[off++] = static_cast<float>(handHashes[me * 4 + r])
                         * indexer_norm_[r];
            } else {
                a[off++] = 0.0f;
            }
        }
    }

    // Sub-blocks. Both sides of the gate (build-time + runtime config)
    // must agree; ObservationLayout already collapses them into the
    // dimensions, so we just mirror the same gating in the writes.
    if constexpr (features::ROUND_SUMMARY) {
        if (rs_cfg_.enabled) write_round_summary(a, bet_history, me);
    }
    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_cfg_.enabled) write_history(a, bet_history, me);
    }

    return obs;
}

void ObservationBuilder::write_round_summary(
    torch::TensorAccessor<float, 1>&     a,
    const std::vector<BetHistoryEntry>&  bet_history,
    int                                  current_player) const
{
    constexpr int R = RoundSummaryConfig::num_rounds;
    constexpr int F = RoundSummaryConfig::feat_per_round;

    // Aggregate bet_history into per-round totals. Cheap — bet_history
    // is bounded by max_raises_per_round × num_rounds × players + calls,
    // a few dozen entries at most.
    uint32_t my_chips[R]      = {};
    uint32_t opp_chips[R]     = {};
    int      raises_in[R]     = {};
    int      last_aggressor[R];
    for (int r = 0; r < R; ++r) last_aggressor[r] = -1;

    for (const auto& e : bet_history) {
        if (e.round >= R) continue;  // defensive
        if (static_cast<int>(e.player) == current_player) my_chips[e.round]  += e.amount;
        else                                              opp_chips[e.round] += e.amount;
        if (e.is_aggressive) {
            raises_in[e.round]++;
            last_aggressor[e.round] = e.player;
        }
    }

    const float raises_norm = static_cast<float>(max_raises_norm_);
    const int   dst_off     = layout_.round_summary_off;
    for (int r = 0; r < R; ++r) {
        const int base = dst_off + r * F;
        a[base + 0] = static_cast<float>(my_chips[r])  / stack_norm_;
        a[base + 1] = static_cast<float>(opp_chips[r]) / stack_norm_;
        a[base + 2] = static_cast<float>(raises_in[r]) / raises_norm;
        a[base + 3] = (last_aggressor[r] == current_player) ? 1.0f : 0.0f;
    }
}

void ObservationBuilder::write_history(
    torch::TensorAccessor<float, 1>&     a,
    const std::vector<BetHistoryEntry>&  bet_history,
    int                                  current_player) const
{
    const int T = hist_cfg_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;

    // Truncate to the most-recent T actions if the hand has run longer.
    // In HU NLHE with max_raises_per_round=4 and 4 rounds this is unlikely
    // to fire, but the env doesn't enforce a hard cap so we guard anyway.
    const int n_total = static_cast<int>(bet_history.size());
    const int n_keep  = std::min(n_total, T);
    const int start   = n_total - n_keep;

    // Mask block: [history_off : history_off + T]
    const int mask_off = layout_.history_off;
    for (int i = 0; i < n_keep; ++i) a[mask_off + i] = 1.0f;
    // (remaining slots already zero from torch::zeros)

    // Token block: [history_off + T : history_off + T + T*F], row-major.
    const int tok_off = layout_.history_off + T;
    for (int i = 0; i < n_keep; ++i) {
        const auto& e = bet_history[start + i];
        const int   base = tok_off + i * F;

        const float amt = static_cast<float>(e.amount);
        a[base + 0] = amt / stack_norm_;
        a[base + 1] = amt / pot_norm_;
        a[base + 2] = (static_cast<int>(e.player) == current_player) ? 1.0f : 0.0f;
        a[base + 3] = e.is_aggressive ? 1.0f : 0.0f;
        // round one-hot
        a[base + 4] = (e.round == 0) ? 1.0f : 0.0f;
        a[base + 5] = (e.round == 1) ? 1.0f : 0.0f;
        a[base + 6] = (e.round == 2) ? 1.0f : 0.0f;
        a[base + 7] = (e.round == 3) ? 1.0f : 0.0f;
    }
}

} // namespace poker_ppo
