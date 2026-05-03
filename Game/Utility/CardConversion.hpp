#pragma once
//
// CardConversion.hpp — bridge between the engine's `deck.h` card encoding
// and Ray Wotton's TwoPlusTwo (2p2) hand-evaluator encoding.
//
// `deck.h` stores cards as
//     id_dh = (rank << 2) | suit_dh        rank ∈ [0, 13), suit ∈ [0, 4)
//   with suit table "shdc" → suit 0=s, 1=h, 2=d, 3=c.
//
// The TwoPlusTwo evaluator (Ray Wotton's tables) uses
//     id_2p2 = (rank * 4) + suit_2p2 + 1
//   with suit order "cdhs" → suit 0=c, 1=d, 2=h, 3=s. So
//     suit_2p2 = 3 - suit_dh
//     id_2p2   = (id_dh & ~3) + (3 - (id_dh & 3)) + 1
//
// Both Transitioner::makeTerminal and the showdown test used to inline
// this conversion locally. Centralising it here ensures any future
// encoding tweak is a single-site fix.
//

#include <cstdint>

namespace Game {

// deck.h card-id (uint8_t, 0..51) → TwoPlusTwo card-id (int, 1..52).
[[nodiscard]] inline constexpr int deck_to_two_plus_two(uint8_t c) noexcept {
    const int rank      = c >> 2;
    const int suit_dh   = c & 3;
    const int suit_2p2  = 3 - suit_dh;
    return rank * 4 + suit_2p2 + 1;
}

}  // namespace Game
