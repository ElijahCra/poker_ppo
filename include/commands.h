#pragma once
//
// CLI command entry points. main.cpp parses argv and dispatches to one
// of these. Each cmd_* owns its lifecycle and returns a Unix exit code.
//

#include "config.h"
#include "environment.h"
#include "ppo.h"

#include <torch/torch.h>

#include <string>

namespace poker_ppo {

// PPO self-play training. Sets up league + BR + metrics, runs to
// kPPOConfig.total_timesteps, saves on exit.
int cmd_train(IPokerEnvironmentFactory& factory,
              const PokerConfig&        poker_cfg,
              torch::Device             device,
              PPOTrainer::Strategy      strategy);

// Interactive REPL over stdin/stdout. Driven by tools/play.py.
int cmd_play(IPokerEnvironmentFactory& factory,
             const BetConfig&          bet_cfg,
             torch::Device             device,
             const std::string&        model_path);

// Rollout-strategy A/B timing. No optimiser update.
int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters);

// Standalone best-response (exploitability) eval of a SAVED model: loads
// the .pt, trains a fresh exploiter against it (kBRConfig; POKER_PPO_BR_SEEDS
// for multi-seed), and prints the bb/hand lower bound. No training of the
// target. The model must have been trained with the current config /
// POKER_PPO_HISTORY_ENCODER so the architecture matches.
int cmd_br_eval(IPokerEnvironmentFactory& factory,
                const PokerConfig&        poker_cfg,
                torch::Device             device,
                const std::string&        model_path);

} // namespace poker_ppo
