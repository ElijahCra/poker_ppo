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
// kPPOConfig.total_timesteps, saves on exit. When resume_dir is non-empty
// the trainer loads the latest checkpoint from <resume_dir>/ckpt/ and
// reopens metrics CSVs in append mode under <resume_dir>/.
int cmd_train(IPokerEnvironmentFactory& factory,
              const PokerConfig&        poker_cfg,
              torch::Device             device,
              PPOTrainer::Strategy      strategy,
              const std::string&        resume_dir = "");

// Interactive REPL over stdin/stdout. Driven by tools/play.py.
int cmd_play(IPokerEnvironmentFactory& factory,
             const BetConfig&          bet_cfg,
             torch::Device             device,
             const std::string&        model_path);

// Rollout-strategy A/B timing. No optimiser update.
int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters);

} // namespace poker_ppo
