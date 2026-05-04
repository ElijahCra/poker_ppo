#pragma once
//
// commands.h — top-level CLI command entry points.
//
// `main.cpp` parses argv into a small struct of options and dispatches to
// one of these. Each cmd_* function owns the lifecycle of its command:
// builds a PPOTrainer (or whatever it needs) from the factory, runs to
// completion, and returns a Unix-style exit code (0 = success).
//

#include "config.h"
#include "environment.h"
#include "ppo.h"

#include <torch/torch.h>

#include <string>

namespace poker_ppo {

// Train the PPO self-play loop end-to-end. Sets up the league + BR
// evaluator + metrics logging, runs for `kPPOConfig.total_timesteps`, and
// saves the trained model on exit.
int cmd_train(IPokerEnvironmentFactory& factory,
              const PokerConfig&        poker_cfg,
              torch::Device             device,
              PPOTrainer::Strategy      strategy);

// Interactive REPL exposing one PokerEnvironment + the trained network
// over stdin/stdout. Driven by `tools/play.py`.
int cmd_play(IPokerEnvironmentFactory& factory,
             const BetConfig&          bet_cfg,
             torch::Device             device,
             const std::string&        model_path);

// A/B-time the rollout strategies. No optimiser update — measures rollout
// throughput only.
int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters);

} // namespace poker_ppo
