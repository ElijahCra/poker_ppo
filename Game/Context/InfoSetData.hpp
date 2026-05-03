#pragma once

#include "HashUtil.hpp"

namespace Game {
// Structured info set data - stores card hashes and bet / call actions per round
// all together equals info available to one player over all rounds (or rounds seen so far)
struct InfoSetData {
    std::array<uint64_t, 4> cardHashes{};  // Card hash for each round (preflop, flop, turn, river)
    std::array<std::vector<uint32_t>, 4> actionsPerRound{};  // Action sequences per round
    int visibleRounds = 0;  // Number of rounds where cards have been revealed

    InfoSetData() noexcept {
        for (auto& seq : actionsPerRound) {
            seq.reserve(Game::MAX_BET_ACTIONS);
        }
    }

    void reset() noexcept {
        cardHashes.fill(0);
        for (auto& seq : actionsPerRound) {
            seq.clear();
        }
        visibleRounds = 0;
    }

    void setCardHash(const int round, const uint64_t hash) noexcept {
        cardHashes[round] = hash;
        visibleRounds = std::max(visibleRounds, round + 1);
    }

    void addAction(const int round, const uint32_t actionValue) noexcept {
        if (round < 4) {
            actionsPerRound[round].push_back(actionValue);
        }
    }

    // Convert action variant to double value
    static uint32_t actionToValue(const Action& action) {
        return std::visit([]<typename ActionType>(const ActionType& a) -> uint32_t {
            using T = std::decay_t<ActionType>;
            if constexpr (std::is_same_v<T, Raise> || std::is_same_v<T, Call>) {
                return a.amount;
            } else if constexpr (std::is_same_v<T, Check>) {
                return 0; // check = no value
            } else {
                __builtin_unreachable();  // tell compiler this path is impossible
            }
        }, action);
    }

    // Generate unique string ID (for compatibility/debugging)
    [[nodiscard]] std::string toStringId() const {
        std::string result;
        for (int r = 0; r < visibleRounds; ++r) {
            if (r > 0) {
                result += "|";
            }
            result += std::to_string(cardHashes[r]);
            for (uint32_t actionVal : actionsPerRound[r]) {
                result += ",";
                result += std::to_string(actionVal);
            }
        }
        return result;
    }

    // Generate numerical unique ID using boost::hash_combine style
    [[nodiscard]] uint64_t toNumericId() const noexcept {
        uint64_t hash = 0;

        for (int r = 0; r < visibleRounds; ++r) {
            // Mix in card hash
            hash = HashUtil::combineHash(hash, cardHashes[r]);

            // Mix in action count for this round
            hash = HashUtil::combineHash(hash, actionsPerRound[r].size());

            // Mix in each action
            for (uint32_t actionVal : actionsPerRound[r]) {
                hash = HashUtil::combineHash(hash, static_cast<uint64_t>(actionVal));
            }
        }
        return hash;
    }

    // Equality comparison for use in hash maps
    bool operator==(const InfoSetData& other) const noexcept {
        if (visibleRounds != other.visibleRounds) return false;
        for (int r = 0; r < visibleRounds; ++r) {
            if (cardHashes[r] != other.cardHashes[r]) return false;
            if (actionsPerRound[r] != other.actionsPerRound[r]) return false;
        }
        return true;
    }
};
}