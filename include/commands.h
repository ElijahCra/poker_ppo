#pragma once

// CLI command entry points

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

// Rollout-strategy A/B timing
int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters);

} // namespace poker_ppo
