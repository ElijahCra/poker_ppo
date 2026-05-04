#include "poker_env.h"

#include "ActionPolicy.hpp"
#include "BettingConfig.hpp"
#include "Context/GameContext.hpp"
#include "GameBase.hpp"

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
      // Norms: stack_norm = initial_stack, pot_norm = 2*initial_stack so
      // both sit in roughly [0, 1]; max_raises_norm guards against div-by-0.
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

    // Reward is scaled separately from the obs so per-hand rewards land
    // in O(0.1) range (≈ utility_in_mbb / 10·big_blind). Avoids both
    // vanishing gradients (stack-normalised) and mode collapse
    // (big_blind-normalised).
    reward_norm_ = 10.0f * static_cast<float>(gcfg.big_blind);

    action_table_.assign(A_, std::nullopt);
}

int PokerEnvironment::current_player() const {
    return game_->getCurrentPlayer();
}

// ── public state accessors ─────────────────────────────────────────────────

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

int PokerEnvironment::obs_dim() const {
    // Defined here for completeness even though the inline override in
    // poker_env.h would suffice — kept for docstring locality.
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

StepResult PokerEnvironment::reset() {
    game_->reInitialize();
    bet_history_.clear();
    auto_advance_chance();
    rebuild_action_table();
    return { observation(), 0.0f, false, compute_mask() };
}

StepResult PokerEnvironment::step(int action_idx) {
    if (action_idx < 0 || action_idx >= A_) {
        throw std::invalid_argument("action index out of range");
    }
    if (!action_table_[action_idx].has_value()) {
        throw std::invalid_argument("agent selected an illegal action");
    }

    const int seat_before  = game_->getCurrentPlayer();
    const int round_before = game_->getContext().getRoundNumber();

    // Record the action into the bet-history sequence *before*
    // transitioning, so player and round reflect the pre-step state.
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
        // Terminal reward for seat 0 in mbb / reward_norm.
        const float r = static_cast<float>(game_->getUtility(0)) / reward_norm_;
        // Obs/mask are unused after terminal; PPO calls reset() next.
        auto obs  = torch::zeros({obs_builder_.obs_dim()});
        auto mask = torch::zeros({A_});
        return { obs, r, true, mask };
    }

    rebuild_action_table();
    return { observation(), 0.0f, false, compute_mask() };
}

// ── internals ────────────────────────────────────────────────────────────────

void PokerEnvironment::auto_advance_chance() {
    while (!game_->isTerminal() && game_->getType() == "chance") {
        game_->transition(::Game::Chance{});
    }
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
            } else {
                // Chance never appears in an action state; ignore.
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
