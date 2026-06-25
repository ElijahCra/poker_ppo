#include "poker_env.h"

#include "ActionPolicy.hpp"
#include "BettingConfig.hpp"
#include "Context/GameContext.hpp"
#include "GameBase.hpp"
#include "GameState.hpp"
#include "Utility/AllInEquity.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace poker_ppo {

PokerEnvironment::PokerEnvironment(const PokerConfig& poker_cfg,
                                   const BetConfig&   bet_cfg,
                                   uint64_t           seed)
    : poker_cfg_(poker_cfg),
      bet_cfg_(bet_cfg),
      rng_(seed),
      game_betting_cfg_(::Game::make_default_betting_config(poker_cfg.game)),
      game_(std::make_unique<::Game::DiscreteGame>(rng_, poker_cfg.game, game_betting_cfg_)),
      // stack_norm = initial_stack, pot_norm = 2*initial_stack so both
      // sit in ~[0, 1]. max_raises_norm guards against div-by-0.
      obs_builder_(
          poker_cfg.hist,
          poker_cfg.round_summary,
          /*stack_norm=*/      static_cast<float>(poker_cfg.game.initial_stack),
          /*pot_norm=*/        2.0f * static_cast<float>(poker_cfg.game.initial_stack),
          /*max_raises_norm=*/ poker_cfg.game.max_raises_per_round)
{
    A_ = poker_cfg_.action_count();
    if (A_ != bet_cfg_.action_count()) {
        throw std::invalid_argument(
            "BetConfig.action_count() must equal PokerConfig.action_count() "
            "(2 + num_raise_slots)");
    }

    bet_history_.reserve(poker_cfg.hist.max_history_len * 2);

    const auto& gcfg = poker_cfg_.game;
    allin_slot_ = gcfg.include_all_in_slot
        ? 2 + static_cast<int>(gcfg.pot_fractions.size())
        : -1;

    // Reward scaling lands per-hand rewards in O(0.1) — avoids vanishing
    // gradients (stack-normalised) and mode collapse (BB-normalised).
    reward_norm_ = 10.0f * static_cast<float>(gcfg.big_blind);

    allin_equity_    = poker_cfg_.allin_equity;
    allin_equity_mc_ = poker_cfg_.allin_equity_mc_samples;
    if (std::getenv("POKER_PPO_NO_ALLIN_EQUITY") != nullptr) {
        allin_equity_ = false;
    }

    action_table_.assign(A_, std::nullopt);
}

int PokerEnvironment::current_player() const {
    return game_->getCurrentPlayer();
}

std::array<int, 2> PokerEnvironment::hole_cards(int player) const {
    auto h = game_->getContext().getHoleCards(player);
    return {static_cast<int>(h[0]), static_cast<int>(h[1])};
}
int PokerEnvironment::community_count() const {
    return game_->getContext().getCommunityCount();
}
int PokerEnvironment::community_card(int idx) const {
    return static_cast<int>(game_->getContext().getCommunityCard(idx));
}
int PokerEnvironment::pot() const {
    return static_cast<int>(game_->getContext().getPot());
}
int PokerEnvironment::current_bet() const {
    return static_cast<int>(game_->getCurrentBet());
}
int PokerEnvironment::stack(int player) const {
    return static_cast<int>(game_->getContext().getStack(player));
}
int PokerEnvironment::round() const {
    return game_->getContext().getRoundNumber();
}
int PokerEnvironment::raise_num() const {
    return game_->getContext().getRaiseNum();
}
int PokerEnvironment::terminal_utility(int player) const {
    return is_terminal()
        ? static_cast<int>(game_->getUtility(player)) : 0;
}

bool PokerEnvironment::is_terminal() const {
    return game_->isTerminal();
}

bool PokerEnvironment::terminal_was_fold() const {
    return game_->isTerminal() &&
           game_->getTerminalReason() == ::Game::TerminalState::FOLD;
}

int PokerEnvironment::obs_dim() const {
    return obs_builder_.obs_dim();
}

torch::Tensor PokerEnvironment::observation() const {
    return obs_builder_.build(
        game_->getContext(), bet_history_,
        static_cast<int>(game_->getCurrentBet()));
}

torch::Tensor PokerEnvironment::legal_action_mask() const {
    return compute_mask();
}

void PokerEnvironment::write_obs_into(float* dst) const {
    obs_builder_.build_into(
        dst, game_->getContext(), bet_history_,
        static_cast<int>(game_->getCurrentBet()));
}

void PokerEnvironment::write_mask_into(float* dst) const {
    std::memset(dst, 0, sizeof(float) * static_cast<size_t>(A_));
    for (int i = 0; i < A_; ++i) {
        if (action_table_[i].has_value()) dst[i] = 1.0f;
    }
}

StepResult PokerEnvironment::reset() {
    auto obs  = torch::empty({obs_builder_.obs_dim()});
    auto mask = torch::empty({A_});
    auto sl   = reset_into(obs.data_ptr<float>(), mask.data_ptr<float>());
    return { obs, sl.reward, sl.done, mask };
}

StepLite PokerEnvironment::reset_into(float* obs_dst, float* mask_dst) {
    game_->reInitialize();
    bet_history_.clear();
    auto_advance_chance();
    rebuild_action_table();
    write_obs_into(obs_dst);
    write_mask_into(mask_dst);
    return {0.0f, false};
}

StepResult PokerEnvironment::step(int action_idx) {
    auto obs  = torch::empty({obs_builder_.obs_dim()});
    auto mask = torch::empty({A_});
    auto sl   = step_into(action_idx, obs.data_ptr<float>(), mask.data_ptr<float>());
    return { obs, sl.reward, sl.done, mask };
}

StepLite PokerEnvironment::step_into(int action_idx,
                                     float* obs_dst, float* mask_dst) {
    if (action_idx < 0 || action_idx >= A_) {
        throw std::invalid_argument("action index out of range");
    }
    if (!action_table_[action_idx].has_value()) {
        throw std::invalid_argument("agent selected an illegal action");
    }

    const int seat_before  = game_->getCurrentPlayer();
    const int round_before = game_->getContext().getRoundNumber();

    // Record into bet_history before transition so player/round reflect
    // the pre-step state.
    {
        BetHistoryEntry e{};
        e.player = static_cast<uint8_t>(seat_before);
        e.round  = static_cast<uint8_t>(round_before);

        std::visit([&]<typename T>(const T& a) {
            using U = std::decay_t<T>;
            if constexpr (std::is_same_v<U, ::Game::Raise>) {
                e.amount        = a.amount;
                e.is_aggressive = true;
            } else if constexpr (std::is_same_v<U, ::Game::Call>) {
                e.amount        = a.amount;
                e.is_aggressive = false;
            } else {  // Fold or Check
                e.amount        = 0;
                e.is_aggressive = false;
            }
        }, *action_table_[action_idx]);

        bet_history_.push_back(e);
    }

    game_->transition(*action_table_[action_idx]);

    if (!game_->isTerminal()) {
        auto_advance_chance();
    }

    if (game_->isTerminal()) {
        // All-in before the river resolves straight to a SHOWDOWN terminal
        // with an incomplete board (equal stacks + matched betting ⇒ both
        // players are all-in, never a covering call). Reward the expected
        // equity over run-outs instead of the realised random outcome.
        const bool allin_showdown =
            allin_equity_ &&
            game_->getTerminalReason() == ::Game::TerminalState::SHOWDOWN &&
            game_->getContext().getCommunityCount() < 5;

        const float util = allin_showdown
            ? allin_equity_utility_p0()
            : static_cast<float>(game_->getUtility(0));
        const float r = util / reward_norm_;

        // Obs/mask unused after terminal; PPO resets next.
        std::memset(obs_dst, 0,
                    sizeof(float) * static_cast<size_t>(obs_builder_.obs_dim()));
        std::memset(mask_dst, 0, sizeof(float) * static_cast<size_t>(A_));
        return { r, true };
    }

    rebuild_action_table();
    write_obs_into(obs_dst);
    write_mask_into(mask_dst);
    return { 0.0f, false };
}

int PokerEnvironment::amount_to_call() const {
    // action_table_[1] is Check (amount 0) or Call (amount to match).
    if (action_table_.size() > 1 && action_table_[1].has_value()) {
        return std::visit([]<typename T>(const T& a) -> int {
            if constexpr (std::is_same_v<std::decay_t<T>, ::Game::Call>) {
                return static_cast<int>(a.amount);
            }
            return 0;  // Check / other
        }, *action_table_[1]);
    }
    return 0;
}

void PokerEnvironment::observation_for_hole(int h0, int h1, float* dst) const {
    const uint8_t hole[2] = {static_cast<uint8_t>(h0), static_cast<uint8_t>(h1)};
    obs_builder_.build_into(dst, game_->getContext(), bet_history_,
                            static_cast<int>(game_->getCurrentBet()), hole);
}

void PokerEnvironment::push_state() {
    state_stack_.push_back(Snapshot{
        game_->snapshotState(), bet_history_, action_table_});
}

void PokerEnvironment::pop_state() {
    TORCH_CHECK(!state_stack_.empty(), "pop_state with empty stack");
    auto& s = state_stack_.back();
    game_->restoreState(s.game);
    bet_history_  = std::move(s.hist);
    action_table_ = std::move(s.table);
    state_stack_.pop_back();
}

void PokerEnvironment::auto_advance_chance() {
    while (!game_->isTerminal() && game_->getType() == "chance") {
        game_->transition(::Game::Chance{});
    }
}

float PokerEnvironment::allin_equity_utility_p0() {
    const auto& ctx   = game_->getContext();
    const auto  h0    = ctx.getHoleCards(0);
    const auto  h1    = ctx.getHoleCards(1);
    const int   count = ctx.getCommunityCount();   // 0 (preflop), 3, or 4

    uint8_t board[5];
    for (int i = 0; i < count; ++i) board[i] = ctx.getCommunityCard(i);

    const double eq0 = ::Game::all_in_equity_p0(
        h0.data(), h1.data(), board, count, rng_, allin_equity_mc_);

    // Mirror Game::getUtility's framing so the equity reward shares a mean
    // with the realised one: EU0 = pot·equity0 − contribution0, where
    // contribution0 = initial_stack − stack(0). Zero-sum by construction.
    const double pot     = static_cast<double>(ctx.getPot());
    const double contrib = static_cast<double>(poker_cfg_.game.initial_stack)
                         - static_cast<double>(ctx.getStack(0));
    return static_cast<float>(eq0 * pot - contrib);
}

void PokerEnvironment::rebuild_action_table() {
    action_table_.assign(A_, std::nullopt);
    if (game_->isTerminal() || game_->getType() != "action") return;

    const auto& ctx    = game_->getContext();
    const uint32_t pot = ctx.getPot();
    const uint32_t cb  = game_->getCurrentBet();

    const auto& pot_fractions = poker_cfg_.game.pot_fractions;
    const auto& available     = game_->getActions();
    for (const auto& act : available) {
        std::visit([&]<typename T>(const T& a) {
            using U = std::decay_t<T>;
            if constexpr (std::is_same_v<U, ::Game::Fold>) {
                action_table_[0] = act;
            } else if constexpr (std::is_same_v<U, ::Game::Check>
                               || std::is_same_v<U, ::Game::Call>) {
                action_table_[1] = act;
            } else if constexpr (std::is_same_v<U, ::Game::Raise>) {
                const uint32_t amt = a.amount;
                bool matched = false;
                for (size_t j = 0; j < pot_fractions.size(); ++j) {
                    const uint32_t expected = cb + static_cast<uint32_t>(
                        pot_fractions[j] * pot);
                    if (expected == amt) {
                        action_table_[2 + j] = act;
                        matched = true;
                        break;
                    }
                }
                if (!matched && allin_slot_ >= 0) {
                    action_table_[allin_slot_] = act;
                }
            }
        }, act);
    }
}

torch::Tensor PokerEnvironment::compute_mask() const {
    auto mask = torch::zeros({A_});
    auto acc  = mask.accessor<float, 1>();
    for (int i = 0; i < A_; ++i) {
        if (action_table_[i].has_value()) acc[i] = 1.0f;
    }
    return mask;
}

PokerEnvironmentFactory::PokerEnvironmentFactory(PokerConfig poker_cfg)
    : poker_cfg_(std::move(poker_cfg)) {}

std::unique_ptr<IPokerEnvironment>
PokerEnvironmentFactory::create(const BetConfig& cfg) {
    const uint64_t idx  = instance_counter_.fetch_add(1);
    const uint64_t seed = poker_cfg_.seed ^ (idx * 0xBF58476D1CE4E5B9ull);
    return std::make_unique<PokerEnvironment>(poker_cfg_, cfg, seed);
}

} // namespace poker_ppo
