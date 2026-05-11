#include "cfv_targets.h"
#include "poker_env.h"

#include "Utility/Utility.hpp"
#include "Utility/CardConversion.hpp"

#include <array>
#include <stdexcept>

namespace poker_ppo {

namespace {

// Lex-ordered combo lookup: combos[(combo_idx)] = {a, b}, 0 <= a < b < 52.
struct ComboTable {
    std::array<std::pair<uint8_t, uint8_t>, kCFVHeadDim> combos;

    constexpr ComboTable() : combos{} {
        int idx = 0;
        for (int a = 0; a < 52; ++a) {
            for (int b = a + 1; b < 52; ++b) {
                combos[idx++] = {static_cast<uint8_t>(a),
                                 static_cast<uint8_t>(b)};
            }
        }
    }
};
inline const ComboTable kComboTable{};

}  // namespace

std::pair<int, int> combo_to_cards(int combo_idx) noexcept {
    const auto p = kComboTable.combos[combo_idx];
    return {static_cast<int>(p.first), static_cast<int>(p.second)};
}

CFVTerminalResult compute_cfv_at_terminal(const PokerEnvironment& env,
                                          int                     player)
{
    if (!env.is_terminal()) {
        throw std::logic_error(
            "compute_cfv_at_terminal: env is not at a terminal state");
    }

    const int opp = 1 - player;

    // ── Gather card identities (deck.h encoding, 0..51) ──────────────────
    const auto opp_hole = env.hole_cards(opp);
    const int opp_a = opp_hole[0];
    const int opp_b = opp_hole[1];

    int board[5];
    for (int i = 0; i < 5; ++i) {
        // community_card reads rawCards[4+i] directly — all 5 are dealt at
        // game init, even at preflop terminals.
        board[i] = env.community_card(i);
    }

    // ── Cards-in-use mask: combos overlapping any of these are invalid ───
    bool used[52] = {false};
    used[opp_a]  = true;
    used[opp_b]  = true;
    for (int i = 0; i < 5; ++i) used[board[i]] = true;

    // ── Pre-convert opp + board to TwoPlusTwo encoding once ──────────────
    const int board_2p2[5] = {
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(board[0])),
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(board[1])),
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(board[2])),
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(board[3])),
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(board[4])),
    };

    // Opp's 7-card strength: one lookup, used for every combo comparison.
    int opp_cards[7] = {
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(opp_a)),
        ::Game::deck_to_two_plus_two(static_cast<uint8_t>(opp_b)),
        board_2p2[0], board_2p2[1], board_2p2[2], board_2p2[3], board_2p2[4],
    };
    const int opp_value = Utility::LookupHandValue(opp_cards);

    // ── Chip math: utilities are pot-based and depend only on contribution ──
    // contribution_p = initial_stack - current_stack(p)
    // pot_total      = contribution_0 + contribution_1
    // win:  +pot_total - contribution_p
    // tie:  +pot_total/2 - contribution_p
    // lose: -contribution_p
    const auto& gcfg     = env.game_config();
    const int initial_st = static_cast<int>(gcfg.initial_stack);
    const int contrib_p  = initial_st - env.stack(player);
    const int pot_total  = (initial_st - env.stack(0)) + (initial_st - env.stack(1));

    // Match poker_env reward scaling: utility / (10 * big_blind).
    // Keeps CFV targets on the same scale as the main V head.
    const float reward_norm = 10.0f * static_cast<float>(gcfg.big_blind);

    const float v_win  = static_cast<float>(pot_total - contrib_p) / reward_norm;
    const float v_tie  = (static_cast<float>(pot_total) * 0.5f
                         - static_cast<float>(contrib_p)) / reward_norm;
    const float v_lose = static_cast<float>(-contrib_p) / reward_norm;

    // ── Per-combo evaluation ─────────────────────────────────────────────
    CFVTerminalResult out{};
    out.target.fill(0.0f);
    out.mask.fill(0.0f);

    for (int combo_idx = 0; combo_idx < kCFVHeadDim; ++combo_idx) {
        const auto [a, b] = kComboTable.combos[combo_idx];
        if (used[a] || used[b]) continue;  // mask stays 0

        int p_cards[7] = {
            ::Game::deck_to_two_plus_two(a),
            ::Game::deck_to_two_plus_two(b),
            board_2p2[0], board_2p2[1], board_2p2[2], board_2p2[3], board_2p2[4],
        };
        const int p_value = Utility::LookupHandValue(p_cards);

        float utility;
        if      (p_value >  opp_value) utility = v_win;
        else if (p_value == opp_value) utility = v_tie;
        else                            utility = v_lose;

        out.target[combo_idx] = utility;
        out.mask[combo_idx]   = 1.0f;
    }

    return out;
}

}  // namespace poker_ppo
