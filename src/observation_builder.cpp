#include "observation_builder.h"

#include "BettingConfig.hpp"
#include "Context/GameContext.hpp"
#include "Utility/HandStrength.hpp"

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


torch::Tensor ObservationBuilder::build(
    const Game::GameContext&             ctx,
    const std::vector<BetHistoryEntry>&  bet_history,
    int                                  current_bet) const
{
    auto obs = torch::zeros({layout_.total_dim});
    auto a   = obs.accessor<float, 1>();

    const int me  = ctx.getCurrentPlayer();
    const int opp = 1 - me;

    // Acting player's hole cards only — never leak the opponent's.
    const auto hole = ctx.getHoleCards(me);
    a[layout_.hole_off + hole[0]] = 1.0f;
    a[layout_.hole_off + hole[1]] = 1.0f;

    const int n_comm = ctx.getCommunityCount();
    for (int i = 0; i < n_comm; ++i) {
        a[layout_.community_off + ctx.getCommunityCard(i)] = 1.0f;
    }

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

    // Per-round hand features. Future-round slots stay 0 so we can't leak
    // unseen community cards. Per-round (see HAND_FEATS_PER_ROUND):
    //   0  category / 9                 made-hand strength
    //   1  flush_draw                   4-of-suit, flush not yet made
    //   2  straight_outs / 8            rank-outs (gutshot vs OESD); 0 preflop
    //   3  alive_5-windows / 4          long-term straight progress (preflop too)
    //   4  overcards / 2                hole cards above max board rank
    if constexpr (features::HAND_STRENGTH) {
        const auto hole       = ctx.getHoleCards(me);
        const int  round_now  = ctx.getRoundNumber();

        // Community cards visible at end of round r.
        constexpr int kCommByRound[4] = {0, 3, 4, 5};

        for (int r = 0; r < 4; ++r) {
            if (r > round_now) {
                for (int k = 0; k < ObservationLayout::HAND_FEATS_PER_ROUND; ++k) {
                    a[off++] = 0.0f;
                }
                continue;
            }

            uint8_t   cards[7];
            uint8_t   community[5];
            cards[0] = hole[0];
            cards[1] = hole[1];
            const int n_comm_r = kCommByRound[r];
            for (int i = 0; i < n_comm_r; ++i) {
                const auto c    = ctx.getCommunityCard(i);
                cards[2 + i]    = c;
                community[i]    = c;
            }
            const int n_total = 2 + n_comm_r;

            const int category = ::Game::hand_category(cards, n_total);
            const int max_suit = ::Game::max_suit_count(cards, n_total);
            const int n_overs  = ::Game::overcard_count(hole.data(), community, n_comm_r);

            a[off++] = static_cast<float>(category)
                     / static_cast<float>(::Game::kHandCategoryCount);

            // Suppress draw features once the made hand exists.
            const bool flush_made = (category == 6 || category == 9);
            a[off++] = (max_suit == 4 && !flush_made) ? 1.0f : 0.0f;

            const bool straight_made = (category == 5 || category == 9);
            const int  s_outs        = straight_made
                ? 0
                : ::Game::straight_draw_outs(cards, n_total);
            a[off++] = std::min(1.0f, static_cast<float>(s_outs) / 8.0f);

            const int s_alive = straight_made
                ? 0
                : ::Game::straight_alive_windows(cards, n_total);
            a[off++] = std::min(1.0f, static_cast<float>(s_alive) / 4.0f);

            a[off++] = static_cast<float>(n_overs) / 2.0f;
        }
    }

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

    // bet_history is bounded by ~max_raises × rounds × players + calls,
    // a few dozen entries.
    uint32_t my_chips[R]      = {};
    uint32_t opp_chips[R]     = {};
    int      raises_in[R]     = {};
    int      last_aggressor[R];
    for (int r = 0; r < R; ++r) last_aggressor[r] = -1;

    for (const auto& e : bet_history) {
        if (e.round >= R) continue;
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

    // Keep the most-recent T. Unlikely to fire in HU NLHE, but the env
    // doesn't enforce a hard cap so guard anyway.
    const int n_total = static_cast<int>(bet_history.size());
    const int n_keep  = std::min(n_total, T);
    const int start   = n_total - n_keep;

    // Mask: [history_off : history_off + T]
    const int mask_off = layout_.history_off;
    for (int i = 0; i < n_keep; ++i) a[mask_off + i] = 1.0f;

    // Tokens: [history_off + T : history_off + T + T*F], row-major.
    const int tok_off = layout_.history_off + T;
    for (int i = 0; i < n_keep; ++i) {
        const auto& e = bet_history[start + i];
        const int   base = tok_off + i * F;

        const float amt = static_cast<float>(e.amount);
        a[base + 0] = amt / stack_norm_;
        a[base + 1] = amt / pot_norm_;
        a[base + 2] = (static_cast<int>(e.player) == current_player) ? 1.0f : 0.0f;
        a[base + 3] = e.is_aggressive ? 1.0f : 0.0f;
        a[base + 4] = (e.round == 0) ? 1.0f : 0.0f;
        a[base + 5] = (e.round == 1) ? 1.0f : 0.0f;
        a[base + 6] = (e.round == 2) ? 1.0f : 0.0f;
        a[base + 7] = (e.round == 3) ? 1.0f : 0.0f;
    }
}

} // namespace poker_ppo
