//
// Created by Elijah Crain on 12/31/25.
//

#ifndef QFR_ROUNDDATA_HPP
#define QFR_ROUNDDATA_HPP

#include "../GameState.hpp"

namespace Game {
// Round data
struct RoundData {
    int number = 0;  // 0=preflop, 1=flop, 2=turn, 3=river
    Action previousAction = Chance{};

    void advance() noexcept {
        ++number;
        previousAction = Chance{};
    }

    void reset() noexcept {
        number = 0;
        previousAction = Chance{};
    }

    [[nodiscard]] bool isPreflop() const noexcept { return number == 0; }
    [[nodiscard]] bool isFlop()    const noexcept { return number == 1; }
    [[nodiscard]] bool isTurn()    const noexcept { return number == 2; }
    [[nodiscard]] bool isRiver()   const noexcept { return number == 3; }
};
}

#endif //QFR_ROUNDDATA_HPP