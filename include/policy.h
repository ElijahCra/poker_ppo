#pragma once
//
// IPolicy: minimal action-selection interface for league head-to-heads.
// Trained networks and rule-based anchors share the same API so the match
// loop doesn't special-case them. Operates on CPU tensors at the call site;
// NetworkPolicy moves to device internally.
//

#include "network.h"

#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>

namespace poker_ppo {

class IPolicy {
public:
    virtual ~IPolicy() = default;
    [[nodiscard]] virtual std::string name() const = 0;

    // obs:  [B, obs_dim]      CPU float
    // mask: [B, action_count] CPU float (1=legal, 0=illegal)
    // returns [B] int64 CPU
    virtual torch::Tensor select_actions(const torch::Tensor& obs,
                                         const torch::Tensor& legal_mask) = 0;
};

// Wraps an ActorCritic. Also used for the "random_init" anchor — a fresh
// ActorCritic with the original orthogonal init, never trained.
class NetworkPolicy : public IPolicy {
public:
    NetworkPolicy(ActorCritic net, torch::Device device, std::string name)
        : net_(std::move(net)), device_(device), name_(std::move(name)) {
        net_->eval();
    }

    [[nodiscard]] std::string name() const override { return name_; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        torch::NoGradGuard ng;
        auto obs_d  = obs.to(device_);
        auto mask_d = legal_mask.to(device_);
        auto ar = net_->get_action(obs_d, mask_d);
        return ar.action.to(torch::kCPU).contiguous();
    }

    ActorCritic& net() { return net_; }

private:
    ActorCritic   net_;
    torch::Device device_;
    std::string   name_;
};

// Uniform over legal actions. The "every model should beat this" baseline.
class UniformPolicy : public IPolicy {
public:
    [[nodiscard]] std::string name() const override { return "uniform"; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        (void)obs;
        return legal_mask.multinomial(/*num_samples=*/1, /*replacement=*/true)
                        .squeeze(-1)
                        .to(torch::kInt64);
    }
};

// Walk priority list; fall back to any legal slot. The env may not expose
// Fold (e.g. BB preflop unraised) so a hard-coded "fold fallback" wedge
// would crash with illegal-action.
inline int64_t first_legal(const torch::TensorAccessor<float, 1>& mask,
                           int64_t A,
                           std::initializer_list<int64_t> priority) {
    for (int64_t k : priority) {
        if (k >= 0 && k < A && mask[k] > 0.5f) return k;
    }
    for (int64_t k = 0; k < A; ++k) {
        if (mask[k] > 0.5f) return k;
    }
    return 0;
}

// Prefer Check/Call → Fold → anything legal.
class AlwaysCallPolicy : public IPolicy {
public:
    [[nodiscard]] std::string name() const override { return "always_call"; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        (void)obs;
        const int64_t B = legal_mask.size(0);
        const int64_t A = legal_mask.size(1);
        auto out = torch::zeros({B}, torch::kInt64);
        auto m   = legal_mask.accessor<float, 2>();
        auto a   = out.accessor<int64_t, 1>();
        for (int64_t i = 0; i < B; ++i) {
            a[i] = first_legal(m[i], A, {1, 0});
        }
        return out;
    }
};

// Smallest legal raise → Check/Call → Fold → anything legal.
class AlwaysRaisePolicy : public IPolicy {
public:
    [[nodiscard]] std::string name() const override { return "always_raise"; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        (void)obs;
        const int64_t B = legal_mask.size(0);
        const int64_t A = legal_mask.size(1);
        auto out = torch::zeros({B}, torch::kInt64);
        auto m   = legal_mask.accessor<float, 2>();
        auto a   = out.accessor<int64_t, 1>();
        for (int64_t i = 0; i < B; ++i) {
            int64_t smallest_raise = -1;
            for (int64_t k = 2; k < A; ++k) {
                if (m[i][k] > 0.5f) { smallest_raise = k; break; }
            }
            a[i] = first_legal(m[i], A, {smallest_raise, 1, 0});
        }
        return out;
    }
};

// Pocket pair → shove (largest legal raise / all-in slot). Else fold/check.
//
// Reads hole one-hot from obs[0:52]. Two cards are paired iff
// card_id/4 matches (card_id = rank*4 + suit). Short-deck variants still
// occupy the same 52-slot space — non-existent cards are just never set.
class PairAllInPolicy : public IPolicy {
public:
    [[nodiscard]] std::string name() const override { return "pair_all_in"; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        const int64_t B = obs.size(0);
        const int64_t A = legal_mask.size(1);
        auto out = torch::zeros({B}, torch::kInt64);
        auto o   = obs.accessor<float, 2>();
        auto m   = legal_mask.accessor<float, 2>();
        auto a   = out.accessor<int64_t, 1>();
        for (int64_t i = 0; i < B; ++i) {
            int hole[2] = {-1, -1};
            int n = 0;
            for (int c = 0; c < 52 && n < 2; ++c) {
                if (o[i][c] > 0.5f) hole[n++] = c;
            }
            const bool pair = (n == 2) && (hole[0] / 4 == hole[1] / 4);
            if (pair) {
                int64_t biggest_raise = -1;
                for (int64_t k = A - 1; k >= 2; --k) {
                    if (m[i][k] > 0.5f) { biggest_raise = k; break; }
                }
                a[i] = first_legal(m[i], A, {biggest_raise, 1, 0});
            } else {
                a[i] = first_legal(m[i], A, {0, 1});
            }
        }
        return out;
    }
};

}  // namespace poker_ppo
