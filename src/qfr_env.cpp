#include "qfr_env.h"

#include "ActionPolicy.hpp"
#include "BettingConfig.hpp"
#include "Context/GameContext.hpp"
#include "GameBase.hpp"

#include <cmath>
#include <stdexcept>

namespace poker_ppo {

namespace {

constexpr int CARD_SLOTS    = 52;
constexpr int FEAT_EXTRA    = 10;  // my_stack + opp_stack + pot + cur_bet + raises + 4 round OH + seat
constexpr int STATIC_OBS    = CARD_SLOTS * 2 + FEAT_EXTRA;

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// QFRPokerEnvironment
// ═════════════════════════════════════════════════════════════════════════════

QFRPokerEnvironment::QFRPokerEnvironment(const QFRConfig& qfr_cfg,
                                         const BetConfig& bet_cfg,
                                         uint64_t seed)
    : qfr_cfg_(qfr_cfg), bet_cfg_(bet_cfg), hist_cfg_(qfr_cfg.hist), rng_(seed)
{
    A_ = qfr_cfg_.action_count();
    if (A_ != bet_cfg_.action_count()) {
        throw std::invalid_argument(
            "BetConfig.action_count() must equal QFRConfig.action_count() "
            "(2 + num_raise_slots)");
    }
    static_obs_dim_ = STATIC_OBS;
    obs_dim_        = hist_cfg_.total_obs_dim(static_obs_dim_);
    bet_history_.reserve(hist_cfg_.max_history_len * 2);

    const auto& gcfg = qfr_cfg_.game;
    allin_slot_ = gcfg.include_all_in_slot
        ? 2 + static_cast<int>(gcfg.pot_fractions.size())
        : -1;

    // Obs features stay scaled by stack so they sit in roughly [0, 1].
    stack_norm_  = static_cast<float>(gcfg.initial_stack);
    pot_norm_    = 2.0f * stack_norm_;
    // Reward is scaled separately so per-hand rewards land in O(0.1) range
    // (≈ utility_in_mbb / 10·big_blind).  Avoids both vanishing gradients
    // (stack-normalised) and mode collapse (big_blind-normalised).
    reward_norm_ = 10.0f * static_cast<float>(gcfg.big_blind);
    max_raises_norm_ = std::max<int>(1, gcfg.max_raises_per_round);

    game_betting_cfg_.config.strategy =
        ::Game::BettingConfig::BetSizeStrategy::FIXED_FRACTIONS;
    game_betting_cfg_.config.potFractions      = gcfg.pot_fractions;
    game_betting_cfg_.config.minBet            = gcfg.min_bet;
    game_betting_cfg_.config.minRaise          = gcfg.min_raise;
    game_betting_cfg_.config.maxRaisesPerRound = gcfg.max_raises_per_round;
    game_betting_cfg_.config.allowAllIn        = gcfg.include_all_in_slot;

    game_ = std::make_unique<::Game::DiscreteGame>(rng_, gcfg, game_betting_cfg_);

    action_table_.assign(A_, std::nullopt);
}

int QFRPokerEnvironment::current_player() const {
    return game_->getCurrentPlayer();
}

bool QFRPokerEnvironment::is_terminal() const {
    return game_->isTerminal();
}

torch::Tensor QFRPokerEnvironment::observation() const {
    return compute_observation();
}

torch::Tensor QFRPokerEnvironment::legal_action_mask() const {
    return compute_mask();
}

StepResult QFRPokerEnvironment::reset() {
    game_->reInitialize();
    bet_history_.clear();
    auto_advance_chance();
    rebuild_action_table();
    return { compute_observation(), 0.0f, false, compute_mask() };
}

StepResult QFRPokerEnvironment::step(int action_idx) {
    if (action_idx < 0 || action_idx >= A_) {
        throw std::invalid_argument("action index out of range");
    }
    if (!action_table_[action_idx].has_value()) {
        throw std::invalid_argument("agent selected an illegal action");
    }

    const int seat_before  = game_->getCurrentPlayer();
    const int round_before = game_->getContext().getRoundNumber();

    // Record the action into the bet-history sequence *before* transitioning,
    // so player and round reflect the state in which the action was taken.
    {
        HistEntry e{};
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
        // Reward for seat 0.  getUtility returns chips in mbb; normalise.
        const float r = static_cast<float>(game_->getUtility(0)) / reward_norm_;
        // obs/mask are unused after a terminal flag (PPO layer calls reset()).
        auto obs  = torch::zeros({obs_dim_});
        auto mask = torch::zeros({A_});
        return { obs, r, true, mask };
    }

    rebuild_action_table();
    return { compute_observation(), 0.0f, false, compute_mask() };
}

// ── internals ────────────────────────────────────────────────────────────────

void QFRPokerEnvironment::auto_advance_chance() {
    while (!game_->isTerminal() && game_->getType() == "chance") {
        game_->transition(::Game::Chance{});
    }
}

void QFRPokerEnvironment::rebuild_action_table() {
    action_table_.assign(A_, std::nullopt);
    if (game_->isTerminal() || game_->getType() != "action") return;

    const auto& ctx    = game_->getContext();
    const uint32_t pot = ctx.getPot();
    const uint32_t cb  = game_->getCurrentBet();

    const auto& pot_fractions = qfr_cfg_.game.pot_fractions;
    const auto& available = game_->getActions();
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
                // Chance never appears in action state; ignore.
            }
        }, act);
    }
}

torch::Tensor QFRPokerEnvironment::compute_mask() const {
    auto mask = torch::zeros({A_});
    auto acc  = mask.accessor<float, 1>();
    for (int i = 0; i < A_; ++i) {
        if (action_table_[i].has_value()) acc[i] = 1.0f;
    }
    return mask;
}

torch::Tensor QFRPokerEnvironment::compute_observation() const {
    auto obs = torch::zeros({obs_dim_});
    auto a   = obs.accessor<float, 1>();

    const auto& ctx = game_->getContext();
    const int me   = ctx.getCurrentPlayer();
    const int opp  = 1 - me;

    // Hole cards (only current player's — never leak the opponent's).
    const auto hole = ctx.getHoleCards(me);
    a[hole[0]] = 1.0f;
    a[hole[1]] = 1.0f;

    // Community cards visible for this round.
    const int n_comm = ctx.getCommunityCount();
    for (int i = 0; i < n_comm; ++i) {
        a[CARD_SLOTS + ctx.getCommunityCard(i)] = 1.0f;
    }

    int off = 2 * CARD_SLOTS;
    a[off++] = static_cast<float>(ctx.getStack(me))  / stack_norm_;
    a[off++] = static_cast<float>(ctx.getStack(opp)) / stack_norm_;
    a[off++] = static_cast<float>(ctx.getPot())      / pot_norm_;
    a[off++] = static_cast<float>(game_->getCurrentBet()) / stack_norm_;
    a[off++] = static_cast<float>(ctx.getRaiseNum())
             / static_cast<float>(max_raises_norm_);

    const int round = ctx.getRoundNumber();
    for (int r = 0; r < 4; ++r) a[off++] = (r == round) ? 1.0f : 0.0f;

    a[off++] = static_cast<float>(me);

    // Bet-history block: [T mask] + [T × F token features].
    off = write_history_block(a, off, me);
    (void)off;  // off is now obs_dim_.

    return obs;
}

int QFRPokerEnvironment::write_history_block(
    torch::TensorAccessor<float, 1>& a, int dst_off, int current_player) const
{
    const int T = hist_cfg_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;

    // Truncate to the most-recent T actions if the hand has run longer.  In
    // heads-up NLHE with max_raises_per_round=4 and 4 rounds this is unlikely
    // to fire, but the env doesn't enforce a hard cap so we guard against it.
    const int n_total = static_cast<int>(bet_history_.size());
    const int n_keep  = std::min(n_total, T);
    const int start   = n_total - n_keep;

    // Mask block: [dst_off : dst_off + T]
    const int mask_off = dst_off;
    for (int i = 0; i < n_keep; ++i) a[mask_off + i] = 1.0f;
    // (remaining slots already zero from torch::zeros)

    // Token block: [dst_off + T : dst_off + T + T*F], row-major (token i, feat f)
    const int tok_off = dst_off + T;
    for (int i = 0; i < n_keep; ++i) {
        const auto& e = bet_history_[start + i];
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

    return dst_off + T + T * F;
}

// ═════════════════════════════════════════════════════════════════════════════
// QFRPokerEnvironmentFactory
// ═════════════════════════════════════════════════════════════════════════════

QFRPokerEnvironmentFactory::QFRPokerEnvironmentFactory(QFRConfig qfr_cfg)
    : qfr_cfg_(std::move(qfr_cfg)) {}

std::unique_ptr<IPokerEnvironment>
QFRPokerEnvironmentFactory::create(const BetConfig& cfg) {
    const uint64_t idx  = instance_counter_.fetch_add(1);
    const uint64_t seed = qfr_cfg_.seed ^ (idx * 0xBF58476D1CE4E5B9ull);
    return std::make_unique<QFRPokerEnvironment>(qfr_cfg_, cfg, seed);
}

} // namespace poker_ppo
