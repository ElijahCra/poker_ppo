//
// Created by Elijah Crain on 10/5/25.
//

#ifndef CFR2_TEXAS_GAME_GAMECONTEXT_HPP
#define CFR2_TEXAS_GAME_GAMECONTEXT_HPP
#include <array>
#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <numeric>
#include <functional>

#include "../GameBase.hpp"
#include "hand_index.h"
#include "InfoSetData.hpp"
#include "PlayerData.hpp"
#include "RoundData.hpp"
#include "CardData.hpp"

namespace Game {

class GameContext {
public:

explicit GameContext(const GameConfig& cfg) : m_cfg(&cfg) {
    cfg.validate();
    players.reset(cfg);
    betting.maxRaises = cfg.max_raises_per_round;
    addBlinds();
}

 void advanceRound() noexcept
{
    round.advance();
    players.currentPlayer = 1;
}

// Betting data
struct BettingData {
    uint32_t pot = 0;
    int raiseNum = 0;
    int maxRaises = 4;                // populated from GameConfig
    std::vector<uint32_t> sequence;   // action amounts in mbb

    BettingData() noexcept {
        sequence.reserve(16);
    }

    [[nodiscard]] bool canRaise() const noexcept {
        return raiseNum < maxRaises;
    }

    void reset() noexcept {
        pot = 0;
        raiseNum = 0;
        sequence.clear();
        // maxRaises preserved across hands; set by GameContext from config.
    }
};

// Utility functions
void reset(std::mt19937& rng) {
    players.reset(*m_cfg);
    betting.reset();
    betting.maxRaises = m_cfg->max_raises_per_round;
    round.reset();
    cards.initialize(rng, *m_cfg);
    addBlinds();
}

[[nodiscard]] const GameConfig& config() const noexcept { return *m_cfg; }

// String infoset format
static std::string actionToString(const Action& action) {
    return std::visit([]<typename ActionType>(const ActionType& a) -> std::string {
        if constexpr (std::is_same_v<std::decay_t<ActionType>, Raise> ||
                      std::is_same_v<std::decay_t<ActionType>, Call>) {
            return std::to_string(a.amount);
        } else if constexpr (std::is_same_v<std::decay_t<ActionType>, Check>) {
            return std::to_string(0);
        } else {
            return std::string(a.name);
        }
    }, action);
}

 void addMoney(const int player, const uint32_t amount, const bool isCall = false) noexcept {
    assert(players.stacks[player] >= amount);
    betting.pot += amount;
    players.stacks[player] -= amount;
    betting.sequence.push_back(amount);
    if (!isCall) {
        betting.raiseNum++;
    }
}
void addBlinds()
{
    const uint32_t sb = m_cfg->small_blind;
    const uint32_t bb = m_cfg->big_blind;
    assert(players.stacks[0] >= bb);
    assert(players.stacks[1] >= sb);
    betting.pot += sb + bb;
    players.stacks[1] -= sb;
    players.stacks[0] -= bb;
}

void setRoundStartingPlayer() noexcept
{
    players.currentPlayer = isPreflop() ? 1 : 0; // sb/button/p1 acts first preflop and second postflop
}

[[nodiscard]] uint32_t getPot() const noexcept
{
    return betting.pot;
}

[[nodiscard]] uint32_t getStack(int player) const noexcept { return players.stacks[player]; }

void updateStack(int player, uint32_t amount) noexcept
{
    assert(amount<0 ? players.stacks[player] >= -amount : true); //check that removing from stack is valid
    players.stacks[player] += amount;
}

void updatePot(uint32_t amount) noexcept {betting.pot += amount;}

[[nodiscard]] uint32_t getCurrentPlayer() const noexcept { return players.currentPlayer; }

void nextPlayer() noexcept {
    players.currentPlayer = 1 - players.currentPlayer;
}

uint64_t getCardHash(const int player, int currentRound) const noexcept
{
    return cards.handHashes[player*4 + currentRound];
}

[[nodiscard]] bool isPreflop() const noexcept { return round.number == 0; }
[[nodiscard]] bool isFlop() const noexcept { return round.number == 1; }
[[nodiscard]] bool isTurn() const noexcept { return round.number == 2; }
[[nodiscard]] bool isRiver() const noexcept { return round.number == 3; }

[[nodiscard]] int getRoundNumber() const noexcept { return round.number; }
[[nodiscard]] int getRaiseNum() const noexcept { return betting.raiseNum; }

[[nodiscard]] std::string getInfoSetString(int player) const {
    return players.getInfoSetString(player);
}
[[nodiscard]] uint64_t getInfoSetNumericId(int player) const noexcept {
    return players.getInfoSetNumericId(player);
}
[[nodiscard]] const InfoSetData& getInfoSetData(int player) const noexcept {
    return players.getInfoSetData(player);
}

[[nodiscard]] std::array<uint8_t, 2> getHoleCards(int player) const noexcept {
    return { cards.rawCards[player * 2], cards.rawCards[player * 2 + 1] };
}

[[nodiscard]] int getCommunityCount() const noexcept {
    switch (round.number) {
        case 0: return 0;
        case 1: return 3;
        case 2: return 4;
        default: return 5;
    }
}

[[nodiscard]] uint8_t getCommunityCard(int idx) const noexcept {
    return cards.rawCards[4 + idx];
}

void initializeCards(std::mt19937& rng) { cards.initialize(rng, *m_cfg); }

void setCardHashFromDeck(int player, int currentRound) noexcept {
    players.setCardHash(player, currentRound,
                        cards.handHashes[player * NUM_ROUNDS + currentRound]);
}

void addInfoSetAction(int player, int currentRound, uint32_t actionValue) noexcept {
    players.addAction(player, currentRound, actionValue);
}


// Game state (public so policies / tests / transitioners can reach them)
PlayerData players;
BettingData betting;
RoundData round;
CardData cards;

private:
const GameConfig* m_cfg;   // non-owning; lifetime managed by Game
};

}  // namespace Game

// Specialization of std::hash for InfoSetData (for use in unordered_map)
template <>
struct std::hash<Game::InfoSetData> {
    std::size_t operator()(const Game::InfoSetData& data) const noexcept {
        return data.toNumericId();
    }
};

#endif  // CFR2_TEXAS_GAME_GAMECONTEXT_HPP