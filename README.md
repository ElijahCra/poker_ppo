# ppo_poker

> Self-play PPO for two-player, zero-sum Texas Hold'em

![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=c%2B%2B&logoColor=white)
![LibTorch](https://img.shields.io/badge/LibTorch-PyTorch%20C%2B%2B-EE4C2C?logo=pytorch&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.18-064F8C?logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

A C++ implementation of **Proximal Policy Optimization** for two-player, zero-sum Texas Hold'em. The trainer runs vectorized self-play across `N` parallel hands, utilizes GAE, and learns through fully independent actor and critic networks with legal-action masking baked into the policy head. The game engine itself is decoupled behind an `IPokerEnvironment` interface, so you can plug in any texas hold 'em variant.

## Highlights

- **PPO + GAE** with clipped surrogate objective, clipped value loss, and per-minibatch advantage normalization
- **Independent actor & critic** — no shared trunk, so value gradients can't pollute the policy ([include/network.h](include/network.h), [src/network.cpp](src/network.cpp))
- **Vectorized self-play** — `N` parallel hands per rollout via [`VectorizedEnv`](include/environment.h:99)
- **Legal-action masking** baked into the policy head (illegal logits → −1e8 before softmax)
- **Zero-sum GAE correction** — value bootstrap and advantage flip sign on every player-switch boundary ([src/rollout_buffer.cpp](src/rollout_buffer.cpp))
- **Magnetic Mirror Descent regularization** ([Sokota et al., ICLR 2023](https://arxiv.org/abs/2206.05825)) — loss adds `kl_coef · KL(π_θ ‖ ρ)`, where ρ is a frozen "magnet" snapshot of the policy refreshed every `magnet_update_every` updates. Drives last-iterate convergence in imperfect-information games; set `kl_coef = 0` to recover vanilla PPO ([include/config.h:101](include/config.h:101))
- **Geometric raise sizing** — discrete `{Fold, Check/Call, Raise₁…Raiseₙ}` action space configured via [`BetConfig`](include/types.h:26)

## Architecture

```
              ┌──────────────────────────────────────────────────────┐
              │                  PPOTrainer.train()                  │
              └──────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
     ┌─────────────────┐                         ┌─────────────────┐   clone every   ┌──────────────────┐
     │  VectorizedEnv  │   ─── actions ──►       │  Actor  (MLP)   │ ── N updates ──►│  MMD magnet  ρ   │
     │  N parallel     │   ◄── obs/mask ───      │  policy logits  │                 │  (frozen actor)  │
     │  IPokerEnv      │                         │  + legal mask   │                 └──────────────────┘
     └─────────────────┘                         └─────────────────┘                          │
              │                                           ▲                                   │
              ▼                                           │                                   │
     ┌─────────────────┐         ┌──────────────────────────────────┐                         │
     │ RolloutBuffer   │         │  Critic (MLP)  ──►  V(s)         │                         │
     │  obs, action,   │         └──────────────────────────────────┘                         │
     │  logπ, reward,  │                         │                                            │
     │  done, value,   │                         ▼                              KL(π_θ ‖ ρ)   │
     │  mask, seat     │                        GAE                                   │       │
     └─────────────────┘                         │                                    │       │
              │                                  └──────────────┬─────────────────────┴───────┘
              └─────────────────────────────────►  PPO update   (4 epochs × 4 minibatches)
```

## Quickstart

### Prerequisites

- A C++23 compiler (Clang 17+, GCC 13+, or MSVC 19.37+)
- CMake ≥ 3.18
- LibTorch installed via pytorch.
  
### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Run the example

```bash
./poker_ppo
```

This trains against the `StubPokerEnvironment` defined in [src/main.cpp](src/main.cpp) for 1,000,000 steps and writes a single `.pt` checkpoint to the working directory (the combined `ActorCritic` module is serialized in one file).

The same binary also exposes:

```bash
./poker_ppo --benchmark [iters]            # A/B-time the serial and threadpool rollout strategies
./poker_ppo --play   <model_path>          # interactive REPL against a saved model
./poker_ppo --strategy {serial|threadpool} # choose the rollout strategy for training
```

## Tools

`tools/` contains Python utilities for monitoring training and playing against trained models.

### Live training plots

```bash
python3 tools/plot_live.py --latest
```

Auto-picks the newest directory under `runs/` and renders a live-refreshing dashboard of policy loss, value loss, entropy, KL divergence, clip fraction, explained variance, and learning rate. Pass an explicit `runs/<timestamp>/` directory as a positional argument to plot a specific run.

### Play against a trained model

```bash
python3 tools/play.py \
    --bin   cmake-build-release/poker_ppo \
    --model cmake-build-release/poker_ppo_model_nlhe_full_52.pt \
    --hands 50 \
    --log   human_actions.jsonl
```

Drives the C++ binary through its `--play` REPL and gives you an interactive table to play hands against the bot. The bot's chosen action and its full action-probability distribution are shown after each move (`--no-show-model` hides them). `--seat {0,1,-1}` picks your seat (`-1` alternates each hand); `--log` appends `(state, action)` pairs to a JSONL file for offline analysis.


Contract notes (full version in [include/environment.h:11](include/environment.h)):

- Each environment instance manages **one hand at a time**; PPO creates `num_envs` of them for parallel rollout.
- Rewards are returned from **player 1's perspective**. The trainer handles the sign flip for the acting seat.
- `legal_action_mask()` is a `[action_count]` float tensor (`1.0` = legal, `0.0` = illegal). Mask all raises once `max_bets_per_round` is reached.
- After a betting round ends inside `step()`, deal community cards internally before the next `observation()` call.

## Action space

`BetConfig` defines a discrete action set:

| Index | Action |
|---|---|
| `0` | Fold |
| `1` | Check / Call |
| `2 … 2 + N − 1` | Raise by `min_raise · ratio^i` |

Example with `num_raise_sizes = 5`, `min_raise = 0.5`, `geometric_ratio = 2.0`:

```
[0] Fold
[1] Check/Call
[2] Raise 0.5×   [3] Raise 1×   [4] Raise 2×   [5] Raise 4×   [6] Raise 8×
```

`max_bets_per_round` caps re-raises per street; the environment masks raise actions once the cap is hit.

## Hyperparameter reference

Defaults from [`PPOConfig`](include/types.h:94):

| Parameter | Default | Notes |
|---|---|---|
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE smoothing |
| `clip_coef` | `0.1` | PPO clipping ε |
| `ent_coef` | `0.05` | Entropy bonus — kept high for imperfect-information exploration |
| `vf_coef` | `0.5` | Value-loss weight |
| `max_grad_norm` | `0.5` | Global gradient clip |
| `clip_vloss` | `true` | Clipped value loss |
| `norm_advantages` | `true` | Per-minibatch advantage normalization |
| `learning_rate` | `2.5e-4` | Adam |
| `anneal_lr` | `true` | Linear LR decay over training |
| `num_envs` | `8` | Parallel self-play games |
| `num_steps` | `128` | Steps per env per rollout |
| `update_epochs` | `4` | Passes over the rollout buffer |
| `num_minibatches` | `4` | Minibatches per epoch |
| `total_timesteps` | `10 000 000` | Training budget |
| `hidden_dim` | `512` | MLP hidden width |
| `num_layers` | `3` | MLP depth (per actor & critic) |

## Training output

Each PPO update emits a `PPOTrainer::UpdateStats` record to the registered log callback ([include/ppo.h:35](include/ppo.h)):

| Field | Meaning |
|---|---|
| `update`, `global_step` | Update index and total env steps so far |
| `policy_loss` | Clipped surrogate loss |
| `value_loss` | (Optionally clipped) value-function MSE |
| `entropy` | Mean policy entropy |
| `approx_kl` | Approximate KL between old and new policy |
| `clip_fraction` | Fraction of samples hit by the PPO clip |
| `explained_variance` | `1 − Var(returns − V) / Var(returns)` |
| `learning_rate` | Current (possibly annealed) LR |

`trainer.save("path.pt")` writes a single file containing the combined `ActorCritic` module (actor tower + critic tower + optional shared history encoder). The Python utility [`tools/plot_live.py`](tools/plot_live.py) renders live training curves from `runs/<timestamp>/`.

## References

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015) — [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (2016) — [arXiv:1603.01121](https://arxiv.org/abs/1603.01121) — motivates the elevated `ent_coef` for IIGs
- Srinivasan, Lanctot, Zambaldi, Pérolat, Tuyls, Munos & Bowling, *Actor-Critic Policy Optimization in Partially Observable Multiagent Environments* (NeurIPS 2018) — [arXiv:1810.09026](https://arxiv.org/abs/1810.09026) — connects policy-gradient methods to regret minimization in imperfect-information games; foundational justification for PPO-style self-play in poker
- Sokota et al., *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games* (ICLR 2023) — [arXiv:2206.05825](https://arxiv.org/abs/2206.05825) — Magnetic Mirror Descent; basis for the `kl_coef · KL(π_θ ‖ ρ)` regularizer



## License

TBD
