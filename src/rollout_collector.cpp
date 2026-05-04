#include "rollout_collector.h"

#include "opponent_manager.h"
#include "rollout_pool.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace poker_ppo {

namespace {

struct BootstrapTensors {
    torch::Tensor values;     // [2, N] float32 CPU
    torch::Tensor terminal;   // [2, N] float32 CPU
};

// V at rollout-end for the GAE bootstrap. Critic-forwards `cur_obs` (from
// the next acting player's view), then uses zero-sum negation for the
// other player's view to fill the [2, N] tensors compute_returns expects.
BootstrapTensors build_bootstrap(
    ActorCritic&                            network,
    const torch::Tensor&                    cur_obs,           // [N, D] device
    const torch::Tensor&                    cur_player_cpu,    // [N] int32 CPU
    const std::vector<PlayerRolloutState>&  player_state)
{
    const int N = static_cast<int>(cur_obs.size(0));
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Critic-only forward; NoGrad is in scope at the call site.
    auto V_carry = network->get_value(cur_obs).detach().to(torch::kCPU).contiguous();

    auto values   = torch::zeros({2, N}, f_cpu);
    auto terminal = torch::zeros({2, N}, f_cpu);

    auto v_acc  = V_carry.accessor<float, 1>();
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();
    auto bv_acc = values.accessor<float, 2>();
    auto bt_acc = terminal.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        const int   next_actor = cp_acc[i];
        const float v          = v_acc[i];
        // V at carry obs is from the next-acting player's perspective; the
        // other player's V is its zero-sum negation.
        bv_acc[next_actor][i]     = v;
        bv_acc[1 - next_actor][i] = -v;
        bt_acc[0][i] = player_state[i].tail_was_terminal[0] ? 1.0f : 0.0f;
        bt_acc[1][i] = player_state[i].tail_was_terminal[1] ? 1.0f : 0.0f;
    }
    return {values, terminal};
}

}  // namespace

RolloutCollector::RolloutCollector(IPokerEnvironmentFactory& factory,
                                   const BetConfig&          bet_cfg,
                                   int                       num_envs,
                                   int                       num_steps,
                                   torch::Device             device)
    : device_(device), num_envs_(num_envs), num_steps_(num_steps)
{
    vec_env_ = std::make_unique<VectorizedEnv>(factory, bet_cfg, num_envs);
    const int obs_dim      = vec_env_->obs_dim();
    const int action_count = vec_env_->action_count();
    buffer_ = std::make_unique<RolloutBuffer>(
        num_steps, num_envs, obs_dim, action_count, device_);
}

RolloutCollector::~RolloutCollector() = default;

void RolloutCollector::ensure_step_pool() {
    if (step_pool_) return;
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int n  = std::min(num_envs_, hw);
    step_pool_   = std::make_unique<StepThreadPool>(n);
}

void RolloutCollector::init_carry() {
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();
    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    carry_obs_            = torch::zeros({num_envs_, D}, f_cpu);
    carry_legal_mask_     = torch::zeros({num_envs_, A}, f_cpu);
    carry_current_player_ = torch::zeros({num_envs_},    i32_cpu);
    carry_done_           = torch::zeros({num_envs_},    f_cpu);

    vec_env_->reset_all();
    for (int i = 0; i < num_envs_; ++i) {
        carry_obs_[i]        = vec_env_->env(i).observation();
        carry_legal_mask_[i] = vec_env_->env(i).legal_action_mask();
        carry_current_player_.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(vec_env_->env(i).current_player());
    }
}

void RolloutCollector::collect(Strategy         strategy,
                               ActorCritic&     network,
                               OpponentManager& opp_mgr,
                               int              update_idx,
                               float            gamma,
                               float            gae_lambda)
{
    network->eval();
    torch::NoGradGuard no_grad;

    if (strategy == Strategy::Threadpool) ensure_step_pool();

    buffer_->clear();
    opp_mgr.prepare_rollout(update_idx);

    auto& envs = vec_env_->envs_mut();
    const int N = num_envs_;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Per-env per-player rollout state. Seeded from carry_done_ so the
    // first transition after a previous-rollout reset carries the
    // new-episode marker.
    std::vector<PlayerRolloutState> player_state(N);
    for (int i = 0; i < N; ++i) {
        const float d = carry_done_.accessor<float, 1>()[i];
        player_state[i].next_done_flag[0] = d;
        player_state[i].next_done_flag[1] = d;
    }

    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    std::vector<uint8_t> did_reset(N, 0);

    for (int step = 0; step < num_steps_; ++step) {
        // Full-batch forward on the calling thread (single inference call
        // per step regardless of strategy).
        auto ar = network->get_action(cur_obs, cur_mask);

        auto actions_cpu    = ar.action.to(torch::kCPU).contiguous();
        auto logp_cpu       = ar.log_prob.to(torch::kCPU).contiguous();
        auto value_cpu      = ar.value.to(torch::kCPU).contiguous();
        auto cur_obs_cpu    = cur_obs.to(torch::kCPU).contiguous();
        auto cur_mask_cpu   = cur_mask.to(torch::kCPU).contiguous();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();

        // Override actions for envs where a pool snapshot is acting as the
        // non-learner seat. No-op when the pool is disabled or empty.
        opp_mgr.apply_action_overrides(
            cur_obs, cur_mask, cur_player_cpu, actions_cpu);

        auto a_acc  = actions_cpu.accessor<int64_t, 1>();
        auto lp_acc = logp_cpu.accessor<float, 1>();
        auto v_acc  = value_cpu.accessor<float, 1>();
        auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto np_acc      = next_player.accessor<int32_t, 1>();

        std::fill(did_reset.begin(), did_reset.end(), 0);

        // Per-env body. Per-i state is disjoint (envs[i], player_state[i],
        // next_*[i], did_reset[i]) so the body is safe to parallelise across
        // i. RolloutBuffer pushes at (player, i) are also disjoint.
        auto step_body = [&](int i) {
            const int acting = cp_acc[i];

            if (opp_mgr.should_record(i, acting)) {
                player_state[i].record_step(
                    acting, i, *buffer_,
                    cur_obs_cpu[i], cur_mask_cpu[i],
                    a_acc[i], lp_acc[i], v_acc[i]);
            }

            auto res = envs[i]->step(static_cast<int>(a_acc[i]));
            player_state[i].step_reward(res.reward);

            if (res.done) {
                player_state[i].flush_on_terminal(i, *buffer_);
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
                did_reset[i] = 1;
            } else {
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        };

        if (strategy == Strategy::Threadpool) {
            step_pool_->parallel_for(N, step_body);
        } else {
            for (int i = 0; i < N; ++i) step_body(i);
        }

        // Re-roll opponent assignments serially: opp_mgr's RNG isn't
        // thread-safe.
        for (int i = 0; i < N; ++i) {
            if (did_reset[i]) opp_mgr.on_episode_terminal(i, update_idx);
        }

        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_player = next_player.to(device_);
    }

    // Drain remaining pending transitions; these are truncations, not
    // terminals — compute_returns will bootstrap them from V(carry_obs).
    for (int i = 0; i < N; ++i) {
        player_state[i].flush_on_rollout_end(i, *buffer_);
    }

    auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
    auto bootstrap = build_bootstrap(network, cur_obs, cur_player_cpu, player_state);
    buffer_->compute_returns(gamma, gae_lambda,
                             bootstrap.values, bootstrap.terminal);

    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_current_player_ = cur_player_cpu;
    // Reset carry_done_ — per-player episode-boundary state is re-seeded
    // at the next rollout's start from this tensor, and we've already
    // flushed the pending transitions whose done flag it would have
    // affected.
    carry_done_.zero_();

    global_step_ += num_envs_ * num_steps_;
}

} // namespace poker_ppo
