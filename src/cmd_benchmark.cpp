#include "commands.h"

#include <iostream>

namespace poker_ppo {

int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters)
{
    std::cout << "\n[benchmark mode] comparing serial / threadpool\n";
    PPOTrainer trainer(factory, device);
    trainer.benchmark_rollouts(iters, /*warmup=*/3);
    return 0;
}

} // namespace poker_ppo
