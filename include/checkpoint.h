#pragma once
//
// Periodic save / resume of full trainer state. One numbered file per
// save under <run_dir>/ckpt/update_<N>.pt. Latest is chosen by max(N).
//
// Bundled in a single torch::serialize archive:
//   "network"     — ActorCritic params + buffers
//   "optimizer"   — Adam state (momentum, second moment)
//   "magnet"      — MMD magnet network (when cfg.kl_coef > 0)
//   "update_idx"  — int64 tensor: which update produced the file
//   "global_step" — int64 tensor: rollout-step counter
//   "has_magnet"  — int32 tensor: 0 or 1
//
// Not saved: opponent pool snapshots, BR exploiter state, env RNG. The
// pool rebuilds via its normal cadence post-resume; BR re-warms naturally;
// env RNG drift is bounded and PPO is robust to it.
//

#include "network.h"

#include <torch/torch.h>

#include <filesystem>
#include <optional>
#include <string>

namespace poker_ppo {

class Checkpoint {
public:
    // Creates dir if missing. dir is typically "<run_dir>/ckpt".
    explicit Checkpoint(std::filesystem::path dir);

    void save(int update_idx, int global_step,
              ActorCritic& network,
              torch::optim::Adam& optimizer,
              ActorCritic& magnet);

    struct LoadedState {
        int update_idx;
        int global_step;
        bool magnet_present;
    };

    // Returns the loaded state on success, nullopt if no checkpoint exists.
    // Mutates network, optimizer (always), and magnet (only when both the
    // saved checkpoint AND the passed magnet are non-empty).
    [[nodiscard]] std::optional<LoadedState> load_latest(
        ActorCritic& network,
        torch::optim::Adam& optimizer,
        ActorCritic& magnet);

    // Delete numbered files past keep_last most recent. No-op if fewer.
    void prune(int keep_last);

    const std::filesystem::path& dir() const noexcept { return dir_; }

private:
    std::filesystem::path dir_;
};

}  // namespace poker_ppo
