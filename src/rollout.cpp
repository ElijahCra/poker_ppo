#include "rollout.h"

#include "opponent_manager.h"

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace poker_ppo {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs,
                             int obs_dim, int action_count,
                             torch::Device device)
    : num_steps_(num_steps), num_envs_(num_envs),
      obs_dim_(obs_dim), action_count_(action_count),
      device_(device)
{
    // Storage on CPU: pushes come from per-env worker threads and per-
    // transition device writes would serialise on CUDA's launch queue.
    // One batched H2D copy in flatten().
    auto cpu_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto cpu_i = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    for (int p = 0; p < 2; ++p) {
        obs_[p]         = torch::zeros({num_steps, num_envs, obs_dim},     cpu_f);
        actions_[p]     = torch::zeros({num_steps, num_envs},              cpu_i);
        log_probs_[p]   = torch::zeros({num_steps, num_envs},              cpu_f);
        rewards_[p]     = torch::zeros({num_steps, num_envs},              cpu_f);
        dones_[p]       = torch::zeros({num_steps, num_envs},              cpu_f);
        values_[p]      = torch::zeros({num_steps, num_envs},              cpu_f);
        vbar_[p]        = torch::zeros({num_steps, num_envs},              cpu_f);
        legal_masks_[p] = torch::zeros({num_steps, num_envs, action_count}, cpu_f);
        advantages_[p]  = torch::zeros({num_steps, num_envs},              cpu_f);
        returns_[p]     = torch::zeros({num_steps, num_envs},              cpu_f);
        counts_[p].assign(num_envs, 0);
    }
}

void RolloutBuffer::clear() {
    for (int p = 0; p < 2; ++p) {
        std::fill(counts_[p].begin(), counts_[p].end(), 0);
    }
}

void RolloutBuffer::push(int player, int env_idx,
                         torch::Tensor obs,
                         int64_t action,
                         float log_prob,
                         float reward,
                         float done,
                         float value,
                         float vbar,
                         torch::Tensor mask) {
    const int t = counts_[player][env_idx]++;

    if (obs.device()  != torch::kCPU) obs  = obs.to(torch::kCPU);
    if (mask.device() != torch::kCPU) mask = mask.to(torch::kCPU);

    // Hot path: raw memcpy into the contiguous [T, N, D] storage. The
    // tensor-assign form (obs_[p][t][e] = obs) builds temporaries and a
    // dispatched copy_ per call — with one push per env-step across all
    // worker threads, that overhead serialises in the dispatcher.
    if (obs.is_contiguous() && mask.is_contiguous()) {
        std::memcpy(obs_[player].data_ptr<float>()
                        + (static_cast<size_t>(t) * num_envs_ + env_idx) * obs_dim_,
                    obs.data_ptr<float>(),
                    sizeof(float) * static_cast<size_t>(obs_dim_));
        std::memcpy(legal_masks_[player].data_ptr<float>()
                        + (static_cast<size_t>(t) * num_envs_ + env_idx) * action_count_,
                    mask.data_ptr<float>(),
                    sizeof(float) * static_cast<size_t>(action_count_));
    } else {
        obs_[player][t][env_idx]         = obs;
        legal_masks_[player][t][env_idx] = mask;
    }
    actions_[player]  .accessor<int64_t, 2>()[t][env_idx] = action;
    log_probs_[player].accessor<float,   2>()[t][env_idx] = log_prob;
    rewards_[player]  .accessor<float,   2>()[t][env_idx] = reward;
    dones_[player]    .accessor<float,   2>()[t][env_idx] = done;
    values_[player]   .accessor<float,   2>()[t][env_idx] = value;
    vbar_[player]     .accessor<float,   2>()[t][env_idx] = vbar;
}

void RolloutBuffer::compute_returns(
    float gamma, float lam,
    const torch::Tensor& bootstrap_values,
    const torch::Tensor& bootstrap_terminal) {

    TORCH_CHECK(bootstrap_values.dim() == 2 &&
                bootstrap_values.size(0) == 2 &&
                bootstrap_values.size(1) == num_envs_,
                "bootstrap_values must be [2, num_envs]");
    TORCH_CHECK(bootstrap_terminal.sizes() == bootstrap_values.sizes(),
                "bootstrap_terminal shape must match bootstrap_values");

    auto bv = bootstrap_values.to(torch::kCPU).contiguous();
    auto bt = bootstrap_terminal.to(torch::kCPU).contiguous();
    auto bv_a = bv.accessor<float, 2>();
    auto bt_a = bt.accessor<float, 2>();

    // Unified GAE / Expected-SARSA(λ) "Q-boosting" trace. In the GAE path
    // values_==vbar_==V(s), so this reduces exactly to GAE. In VRPO,
    // values_=Q(s,a), vbar_=V̄(s); the bootstrap term uses V̄ (an exact
    // expectation over the next action distribution — no sampling noise):
    //   δ⁺_t = r_t + γ·V̄(s_{t+1}) − Q(s_t,a_t)
    //   Â_t  = Q(s_t,a_t) − V̄(s_t) + Σ_{k≥0}(λγ)^k δ⁺_{t+k}
    //        = Q_target_t − V̄(s_t),   Q_target_t = Q(s_t,a_t) + trace
    for (int p = 0; p < 2; ++p) {
        advantages_[p].zero_();
        returns_[p].zero_();

        auto rewards_a = rewards_[p].accessor<float, 2>();
        auto dones_a   = dones_[p].accessor<float, 2>();
        auto values_a  = values_[p].accessor<float, 2>();  // Q(s,a) / V(s)
        auto vbar_a    = vbar_[p].accessor<float, 2>();     // V̄(s)   / V(s)
        auto advs_a    = advantages_[p].accessor<float, 2>();
        auto rets_a    = returns_[p].accessor<float, 2>();

        for (int e = 0; e < num_envs_; ++e) {
            const int T = counts_[p][e];
            if (T == 0) continue;
            float lasttrace = 0.0f;
            for (int t = T - 1; t >= 0; --t) {
                float next_nonterminal, next_vbar;
                if (t == T - 1) {
                    if (bt_a[p][e] > 0.5f) {
                        // Terminal: V̄_next = 0.
                        next_nonterminal = 0.0f;
                        next_vbar        = 0.0f;
                    } else {
                        // Truncation: bootstrap V̄(carry_obs).
                        next_nonterminal = 1.0f;
                        next_vbar        = bv_a[p][e];
                    }
                } else {
                    next_nonterminal = 1.0f - dones_a[t + 1][e];
                    next_vbar        = vbar_a[t + 1][e];
                }
                const float delta =
                    rewards_a[t][e] + gamma * next_vbar * next_nonterminal
                    - values_a[t][e];
                lasttrace = delta + gamma * lam * next_nonterminal * lasttrace;
                const float q_target = values_a[t][e] + lasttrace;
                rets_a[t][e] = q_target;
                advs_a[t][e] = q_target - vbar_a[t][e];
            }
        }
    }
}

RolloutBuffer::FlatBatch RolloutBuffer::flatten() const {
    std::vector<torch::Tensor> obs_list, actions_list, logp_list;
    std::vector<torch::Tensor> advs_list, rets_list, vals_list, masks_list;

    for (int p = 0; p < 2; ++p) {
        for (int e = 0; e < num_envs_; ++e) {
            const int T = counts_[p][e];
            if (T == 0) continue;
            obs_list    .push_back(obs_[p]        .slice(0, 0, T).select(1, e));
            actions_list.push_back(actions_[p]    .slice(0, 0, T).select(1, e));
            logp_list   .push_back(log_probs_[p]  .slice(0, 0, T).select(1, e));
            advs_list   .push_back(advantages_[p] .slice(0, 0, T).select(1, e));
            rets_list   .push_back(returns_[p]    .slice(0, 0, T).select(1, e));
            vals_list   .push_back(values_[p]     .slice(0, 0, T).select(1, e));
            masks_list  .push_back(legal_masks_[p].slice(0, 0, T).select(1, e));
        }
    }

    auto to_dev = [this](torch::Tensor t) {
        return t.to(device_, /*non_blocking=*/false, /*copy=*/false);
    };

    if (obs_list.empty()) {
        auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        auto i = torch::TensorOptions().dtype(torch::kInt64).device(device_);
        return {
            torch::zeros({0, obs_dim_},      f),
            torch::zeros({0},                i),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0, action_count_}, f),
        };
    }
    return {
        to_dev(torch::cat(obs_list,     0)),
        to_dev(torch::cat(actions_list, 0)),
        to_dev(torch::cat(logp_list,    0)),
        to_dev(torch::cat(advs_list,    0)),
        to_dev(torch::cat(rets_list,    0)),
        to_dev(torch::cat(vals_list,    0)),
        to_dev(torch::cat(masks_list,   0)),
    };
}

namespace {

// Phase stopwatch for the rollout loop. When `active`, lap() syncs the
// CUDA stream so async kernels are attributed to the phase that launched
// them, then accumulates elapsed time into the target counter. Inert
// (no clock reads, no syncs) when profiling is off, so the normal path
// pays only a predicted-not-taken branch per phase.
struct PhaseTimer {
    bool active;
    bool cuda;
    std::chrono::steady_clock::time_point last;

    PhaseTimer(bool active_, bool cuda_) : active(active_), cuda(cuda_) {
        if (active) last = std::chrono::steady_clock::now();
    }
    void reset() {
        if (active) last = std::chrono::steady_clock::now();
    }
    void lap(double& acc) {
        if (!active) return;
        if (cuda) torch::cuda::synchronize();
        const auto t = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t - last).count();
        last = t;
    }
};

void print_profile(const char* name, const RolloutProfile& p) {
    if (p.rollouts == 0) return;
    const double n = static_cast<double>(p.rollouts);
    const double tot = p.total_ms;
    auto row = [&](const char* phase, double ms) {
        std::cout << "  " << std::setw(16) << std::left << phase << std::right
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << (ms / n) << " ms/rollout"
                  << std::setw(8) << std::setprecision(1)
                  << (tot > 0 ? 100.0 * ms / tot : 0.0) << "%\n";
    };
    std::cout << "\n────────── rollout phase profile [" << name << "] ──────────\n"
              << "  rollouts=" << p.rollouts
              << "  total=" << std::fixed << std::setprecision(2) << (tot / n)
              << " ms/rollout\n  ────────────────────────────────────────────\n";
    row("infer",        p.infer_ms);
    row("d2h",          p.d2h_ms);
    row("overrides",    p.overrides_ms);
    row("alloc_next",   p.alloc_next_ms);
    row("env_step",     p.env_step_ms);
    row("terminal_rng", p.terminal_rng_ms);
    row("h2d",          p.h2d_ms);
    row("returns",      p.returns_ms);
    std::cout << "  ────────────────────────────────────────────\n";
    std::cout.unsetf(std::ios::fixed);
}

// {action, log_prob, value, v_bar[, magnet_logp]} as one [4|5, B] fp32
// tensor — a single D2H round trip per step instead of several. Action
// indices are < the action count (≪ 2^24), so the float32 round trip is
// exact. The explicit fp32 casts are no-ops in eager fp32 but required
// under bf16 autocast capture (critic outputs come back bf16 there).
// magnet_logp (R-NaD) is the magnet's log-prob of the taken action; pass
// an undefined tensor when the transform is off.
torch::Tensor pack_action_result(const ActorCriticImpl::ActionResult& ar,
                                 const torch::Tensor& magnet_logp = {}) {
    std::vector<torch::Tensor> rows = {ar.action.to(torch::kFloat32),
                                       ar.log_prob.to(torch::kFloat32),
                                       ar.value.to(torch::kFloat32),
                                       ar.v_bar.to(torch::kFloat32)};
    if (magnet_logp.defined()) {
        rows.push_back(magnet_logp.to(torch::kFloat32));
    }
    return torch::stack(rows, 0);
}

struct BootstrapTensors {
    torch::Tensor values;     // [2, N] float32 CPU
    torch::Tensor terminal;   // [2, N] float32 CPU
};

// Critic-forwards cur_obs (next acting player's view), then zero-sum
// negates for the other player to fill compute_returns' [2, N] tensors.
// VRPO uses the masked expected value V̄(carry); GAE uses V(carry).
BootstrapTensors build_bootstrap(
    ActorCritic&                            network,
    const torch::Tensor&                    cur_obs,           // [N, D] device
    const torch::Tensor&                    cur_mask,          // [N, A] device
    const torch::Tensor&                    cur_player_cpu,    // [N] int32 CPU
    const std::vector<PlayerRolloutState>&  player_state)
{
    const int N = static_cast<int>(cur_obs.size(0));
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // NoGrad is in scope at the call site.
    auto V_carry = network->get_state_value(cur_obs, cur_mask)
                       .detach().to(torch::kCPU).contiguous();

    auto values   = torch::zeros({2, N}, f_cpu);
    auto terminal = torch::zeros({2, N}, f_cpu);

    auto v_acc  = V_carry.accessor<float, 1>();
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();
    auto bv_acc = values.accessor<float, 2>();
    auto bt_acc = terminal.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        const int   next_actor = cp_acc[i];
        const float v          = v_acc[i];
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

    profiling_ = std::getenv("POKER_PPO_PROFILE") != nullptr;
}

RolloutCollector::~RolloutCollector() {
    if (!profiling_) return;
    print_profile("serial",     prof_[0]);
    print_profile("threadpool", prof_[1]);
}

void RolloutCollector::ensure_step_pool() {
    if (step_pool_) return;
    // Default: one worker per core (capped at num_envs). env_step is the
    // CPU-bound rollout phase (game engine + obs build), parallelised
    // across envs by this pool — so it scales with cores up to a knee,
    // past which per-step sync overhead dominates (each env step is tiny).
    // POKER_PPO_STEP_THREADS overrides the worker count to sweep that knee
    // on a new CPU without a rebuild.
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    int n = std::min(num_envs_, hw);
    if (const char* e = std::getenv("POKER_PPO_STEP_THREADS")) {
        const int v = std::atoi(e);
        if (v > 0) {
            n = std::min(num_envs_, v);
            std::cout << "[rollout] step-pool workers=" << n
                      << " (POKER_PPO_STEP_THREADS)\n";
        }
    }
    step_pool_   = std::make_unique<StepThreadPool>(n);
}

void RolloutCollector::set_rnad(ActorCritic* magnet, float eta) {
    TORCH_CHECK(graph_state_ == GraphState::Unset,
                "set_rnad must be called before the first collect() — the "
                "CUDA graph capture bakes in the magnet forward");
    rnad_magnet_ = magnet;
    rnad_eta_    = eta;
}

namespace {
// Magnet log-prob of the taken action — the logρ side of R-NaD's
// −η(logπ−logρ) reward perturbation. Runs under the caller's NoGrad.
torch::Tensor magnet_logp_of(ActorCritic& magnet,
                             const torch::Tensor& obs,
                             const torch::Tensor& mask,
                             const torch::Tensor& action) {
    return magnet->masked_log_probs(obs, mask)
        .gather(-1, action.unsqueeze(-1))
        .squeeze(-1);
}
}  // namespace

bool RolloutCollector::ensure_cuda_graph(ActorCritic& network) {
    if (!device_.is_cuda()) return false;
    if (std::getenv("POKER_PPO_NO_CUDA_GRAPH") != nullptr) return false;
    if (graph_state_ == GraphState::Failed) return false;
    if (graph_state_ == GraphState::Ready) {
        // One capture per network; pool snapshots / exploiters run eager.
        return graph_net_ == static_cast<const void*>(network.get());
    }

    graph_net_ = network.get();
    try {
        const int N = num_envs_;
        const int D = vec_env_->obs_dim();
        const int A = vec_env_->action_count();
        auto f_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

        // Static inputs, allocated from the regular pool before capture.
        // All-ones mask = every action legal; shapes are all capture needs.
        g_obs_  = torch::zeros({N, D}, f_dev);
        g_mask_ = torch::ones({N, A}, f_dev);

        // bf16 autocast inside the graph: same numerics regime the update
        // already runs in. The autocast CACHE must be off during capture —
        // a cached bf16 weight cast from warmup would be baked in as a
        // stale constant, while uncached casts are recorded as kernels
        // that re-read the live fp32 weights on every replay (the
        // torch.cuda.make_graphed_callables rule).
        const bool graph_amp =
            std::getenv("POKER_PPO_NO_GRAPH_AMP") == nullptr;
        auto set_amp = [](bool on) {
            at::autocast::set_autocast_enabled(at::kCUDA, on);
            if (on) at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
            at::autocast::set_autocast_cache_enabled(!on);
        };

        // Capture must run on a non-default stream; warmup on the same
        // stream first so cuBLAS handles + workspaces exist before the
        // capture window (allocating them inside capture is illegal).
        // One inference pass: policy get_action, plus (R-NaD) the magnet's
        // log-prob of the sampled action. The magnet's parameter storage is
        // stable (in-place refresh), so capturing it is refresh-safe.
        auto run_inference = [&] {
            auto ar = network->get_action(g_obs_, g_mask_);
            torch::Tensor mlp;
            if (rnad_magnet_) {
                mlp = magnet_logp_of(*rnad_magnet_, g_obs_, g_mask_, ar.action);
            }
            return pack_action_result(ar, mlp);
        };

        auto stream = at::cuda::getStreamFromPool();
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            if (graph_amp) set_amp(true);
            for (int i = 0; i < 3; ++i) {
                (void)run_inference();
            }
            at::cuda::getCurrentCUDAStream().synchronize();

            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            // Packed output storage lives in the graph's private pool;
            // each replay rewrites it in place.
            g_packed_ = run_inference();
            graph_->capture_end();
            if (graph_amp) set_amp(false);
        }
        torch::cuda::synchronize();
        graph_state_ = GraphState::Ready;
        std::cout << "[rollout] CUDA-graphed inference forward "
                  << "(POKER_PPO_NO_CUDA_GRAPH=1 disables)\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[rollout] CUDA graph capture failed, eager fallback: "
                  << e.what() << "\n";
        // Capture may have died inside the autocast window — restore.
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::set_autocast_cache_enabled(true);
        graph_.reset();
        g_obs_ = g_mask_ = g_packed_ = torch::Tensor();
        graph_state_ = GraphState::Failed;
        return false;
    }
}

void RolloutCollector::init_carry() {
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();
    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto f_pin   = f_cpu.pinned_memory(device_.is_cuda());

    carry_obs_            = torch::zeros({num_envs_, D}, f_pin);
    carry_legal_mask_     = torch::zeros({num_envs_, A}, f_pin);
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
                               float            gamma,
                               float            gae_lambda)
{
    network->eval();
    // NoGradGuard, not InferenceMode: rollout tensors land in the buffer
    // and feed update()'s autograd forward, but inference tensors can't
    // be inputs to grad-tracked ops in normal mode.
    torch::NoGradGuard no_grad;

    if (strategy == Strategy::Threadpool) ensure_step_pool();
    const bool use_graph = ensure_cuda_graph(network);

    // Opt-in phase profiling. `prof` aggregates across rollouts for this
    // strategy; `total` brackets the whole collect() call.
    auto&      prof = prof_[strategy == Strategy::Threadpool ? 1 : 0];
    const bool cuda = device_.is_cuda();
    PhaseTimer total(profiling_, cuda);

    buffer_->clear();
    opp_mgr.prepare_rollout(global_step_);

    auto& envs = vec_env_->envs_mut();
    const int N = num_envs_;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    // Pinned staging for the per-step obs/mask: H2D from pinned memory is a
    // single DMA instead of a pageable→staging→device double copy. The
    // caching host allocator makes the per-step alloc cheap after warmup,
    // and tracks the in-flight non_blocking upload before reusing a block.
    auto f_pin = torch::TensorOptions().dtype(torch::kFloat32)
                     .device(torch::kCPU).pinned_memory(device_.is_cuda());

    // Seed from carry_done_ so the first transition after a previous-
    // rollout reset carries the new-episode marker.
    std::vector<PlayerRolloutState> player_state(N);
    for (int i = 0; i < N; ++i) {
        const float d = carry_done_.accessor<float, 1>()[i];
        player_state[i].next_done_flag[0] = d;
        player_state[i].next_done_flag[1] = d;
    }

    // Carried state lives on the CPU; it's the source of truth. Each step
    // uploads obs+mask to the device for the forward and downloads only what
    // the forward produces (action/log_prob/value). current_player never
    // feeds a device kernel, so it stays on the CPU — the old code uploaded
    // it and obs/mask only to download them again the next step, a pure
    // host↔device round trip.
    auto cur_obs_cpu    = carry_obs_;             // [N, D]   CPU
    auto cur_mask_cpu   = carry_legal_mask_;      // [N, A]   CPU
    auto cur_player_cpu = carry_current_player_;  // [N] int32 CPU

    torch::Tensor cur_obs, cur_mask;  // device mirrors, refreshed each step

    std::vector<uint8_t> did_reset(N, 0);

    // Pinned staging for the packed step outputs (reused across steps;
    // fully consumed before the next step overwrites it).
    if (!packed_pin_.defined()) {
        packed_pin_ = torch::zeros({rnad_magnet_ ? 5 : 4, N}, f_pin);
    }

    for (int step = 0; step < num_steps_; ++step) {
        PhaseTimer pt(profiling_, cuda);

        // One full-batch forward per step regardless of strategy. Graph
        // path: refill the static buffers, replay; the packed output lands
        // in g_packed_. Eager path: plain upload + forward + pack.
        // non_blocking is safe either way: the forward is queued on the
        // same stream, and cur_obs_cpu / cur_mask_cpu stay alive and
        // unwritten for the rest of the step.
        torch::Tensor packed_dev;
        if (use_graph) {
            g_obs_ .copy_(cur_obs_cpu,  /*non_blocking=*/true);
            g_mask_.copy_(cur_mask_cpu, /*non_blocking=*/true);
            cur_obs  = g_obs_;
            cur_mask = g_mask_;
            pt.lap(prof.h2d_ms);

            graph_->replay();
            packed_dev = g_packed_;
            pt.lap(prof.infer_ms);
        } else {
            cur_obs  = cur_obs_cpu.to(device_, /*non_blocking=*/true);
            cur_mask = cur_mask_cpu.to(device_, /*non_blocking=*/true);
            pt.lap(prof.h2d_ms);

            auto ar = network->get_action(cur_obs, cur_mask);
            torch::Tensor mlp;
            if (rnad_magnet_) {
                mlp = magnet_logp_of(*rnad_magnet_, cur_obs, cur_mask, ar.action);
            }
            packed_dev = pack_action_result(ar, mlp);
            pt.lap(prof.infer_ms);
        }

        // One packed D2H round trip for everything the step needs back on
        // the host. The blocking copy_ into pinned staging is the step's
        // single sync point. The override API mutates an int64 tensor in
        // place, so peel actions off as a (cheap, host-side) cast.
        packed_pin_.copy_(packed_dev, /*non_blocking=*/false);
        auto actions_cpu = packed_pin_[0].to(torch::kInt64);
        pt.lap(prof.d2h_ms);

        opp_mgr.apply_action_overrides(
            cur_obs, cur_mask, cur_player_cpu, actions_cpu);
        pt.lap(prof.overrides_ms);

        auto a_acc  = actions_cpu.accessor<int64_t, 1>();
        auto pk_acc = packed_pin_.accessor<float, 2>();  // [4,N]: a, logp, Q, V̄
        auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

        // empty, not zeros: step_into/reset_into fully writes every row.
        auto next_obs    = torch::empty({N, D}, f_pin);
        auto next_mask   = torch::empty({N, A}, f_pin);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto np_acc      = next_player.accessor<int32_t, 1>();
        float* const next_obs_p  = next_obs.data_ptr<float>();
        float* const next_mask_p = next_mask.data_ptr<float>();

        std::fill(did_reset.begin(), did_reset.end(), 0);
        pt.lap(prof.alloc_next_ms);

        // Per-i state is disjoint, so step_body is safe to parallelise
        // across i. Buffer pushes at (player, i) are also disjoint.
        // step_into/reset_into write obs/mask straight into this env's
        // pinned staging row — the tensor-returning step() allocates
        // several tensors per call, and with one call per env-step across
        // all worker threads those allocs serialise in the CPU allocator.
        auto step_body = [&](int i) {
            const int acting = cp_acc[i];

            if (opp_mgr.should_record(i, acting)) {
                player_state[i].record_step(
                    acting, i, *buffer_,
                    cur_obs_cpu[i], cur_mask_cpu[i],
                    a_acc[i], pk_acc[1][i], pk_acc[2][i], pk_acc[3][i]);

                // R-NaD: perturb rewards zero-sum for this π-controlled
                // action — actor −η(logπ−logρ), opponent +η(logπ−logρ).
                // Lands in the accumulator, i.e. in THIS transition's
                // reward (closed at the actor's next action / terminal),
                // and flows into returns so the critic learns the
                // regularised game. Pool-snapshot actions (should_record
                // false) are not transformed.
                if (rnad_eta_ > 0.0f) {
                    const float d = pk_acc[1][i] - pk_acc[4][i];
                    player_state[i].step_reward(
                        acting == 0 ? -rnad_eta_ * d : rnad_eta_ * d);
                }
            }

            float* const obs_dst  = next_obs_p  + static_cast<size_t>(i) * D;
            float* const mask_dst = next_mask_p + static_cast<size_t>(i) * A;

            const auto sl = envs[i]->step_into(static_cast<int>(a_acc[i]),
                                               obs_dst, mask_dst);
            player_state[i].step_reward(sl.reward);

            if (sl.done) {
                player_state[i].flush_on_terminal(i, *buffer_);
                envs[i]->reset_into(obs_dst, mask_dst);
                did_reset[i] = 1;
            }
            np_acc[i] = static_cast<int32_t>(envs[i]->current_player());
        };

        if (strategy == Strategy::Threadpool) {
            step_pool_->parallel_for(N, step_body);
        } else {
            for (int i = 0; i < N; ++i) step_body(i);
        }
        pt.lap(prof.env_step_ms);

        // Serial: opp_mgr's RNG isn't thread-safe.
        for (int i = 0; i < N; ++i) {
            if (did_reset[i]) opp_mgr.on_episode_terminal(i, global_step_);
        }
        pt.lap(prof.terminal_rng_ms);

        // Hand next state to the carried CPU buffers. These are freshly
        // allocated (no aliasing with what step_body just wrote), so moving
        // the handles is safe; the device upload happens at the top of the
        // next iteration.
        cur_obs_cpu    = std::move(next_obs);
        cur_mask_cpu   = std::move(next_mask);
        cur_player_cpu = std::move(next_player);
    }

    total.reset();  // bracket the returns/bootstrap tail separately

    // Truncations, not terminals — bootstrapped from V(carry_obs).
    for (int i = 0; i < N; ++i) {
        player_state[i].flush_on_rollout_end(i, *buffer_);
    }

    // Bootstrap V̄(carry_obs) needs the final state + mask on the device;
    // upload the carried CPU tensors once (current_player stays on the CPU).
    cur_obs  = cur_obs_cpu.to(device_, /*non_blocking=*/true);
    cur_mask = cur_mask_cpu.to(device_, /*non_blocking=*/true);
    auto bootstrap = build_bootstrap(network, cur_obs, cur_mask,
                                     cur_player_cpu, player_state);
    buffer_->compute_returns(gamma, gae_lambda,
                             bootstrap.values, bootstrap.terminal);

    carry_obs_            = cur_obs_cpu;
    carry_legal_mask_     = cur_mask_cpu;
    carry_current_player_ = cur_player_cpu;
    // Pending transitions are already flushed; clear the boundary state.
    carry_done_.zero_();

    total.lap(prof.returns_ms);

    if (profiling_) {
        prof.total_ms = prof.infer_ms + prof.d2h_ms + prof.overrides_ms
                      + prof.alloc_next_ms + prof.env_step_ms
                      + prof.terminal_rng_ms + prof.h2d_ms + prof.returns_ms;
        ++prof.rollouts;
    }

    global_step_ += num_envs_ * num_steps_;
}

} // namespace poker_ppo
