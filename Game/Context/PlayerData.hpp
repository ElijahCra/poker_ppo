//
// Created by Elijah Crain on 12/31/25.
//

#ifndef QFR_PLAYERDATA_HPP
#define QFR_PLAYERDATA_HPP

namespace Game {

// Player data
struct PlayerData {
    int currentPlayer = 0;  // Current player index
    std::array<uint32_t, NUM_PLAYERS> stacks{};   // populated by reset(cfg)
    std::array<InfoSetData, NUM_PLAYERS> infoSets{};  // Structured info sets for each player



    // Add card hash for a specific round
    void setCardHash(int player, int current_round, uint64_t hash) noexcept {
        infoSets[player].setCardHash(current_round, hash);
    }

    // Add action to current round
    void addAction(int player, int current_round, uint32_t actionValue) noexcept {
        infoSets[player].addAction(current_round, actionValue);
    }

    // Add action from Action variant
    void addAction(int player, int current_round, const Action& action) noexcept {
        infoSets[player].addAction(current_round, InfoSetData::actionToValue(action));
    }

    // Get string representation of info set
    [[nodiscard]] std::string getInfoSetString(int player) const {
        return infoSets[player].toStringId();
    }

    // Get numerical ID of info set
    [[nodiscard]] uint64_t getInfoSetNumericId(int player) const noexcept {
        return infoSets[player].toNumericId();
    }

    // Get direct access to info set data
    [[nodiscard]] const InfoSetData& getInfoSetData(int player) const noexcept {
        return infoSets[player];
    }

    InfoSetData& getInfoSetDataMutable(int player) noexcept {
        return infoSets[player];
    }

    void reset(const GameConfig& cfg) noexcept {
        currentPlayer = 0;
        stacks.fill(cfg.initial_stack);
        for (auto& infoSet : infoSets) {
            infoSet.reset();
        }
    }
};

}

#endif //QFR_PLAYERDATA_HPP