#pragma once
//
// policy.h — minimal action-selection interface used by the league for
//            head-to-head matches.
//
// Both trained ActorCritic networks and rule-based "anchor" policies (always-
// call, always-raise, pair-caller, uniform-random) implement the same IPolicy
// interface so the match-playing loop doesn't need to special-case them.
//
// IPolicy works on CPU tensors at the call site.  NetworkPolicy is responsible
// for moving the batch onto its device and back; rule-based policies do their
// work directly on CPU.
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

    /// Pick an action for each row of the batch.
    ///   obs  : [B, obs_dim]      CPU float
    ///   mask : [B, action_count] CPU float (1 = legal, 0 = illegal)
    ///   returns [B] int64 CPU
    virtual torch::Tensor select_actions(const torch::Tensor& obs,
                                         const torch::Tensor& legal_mask) = 0;
};

// NetworkPolicy — wraps an ActorCritic.
//
// Moves the batch to the network's device, runs masked-softmax sampling, and
// returns the actions on CPU.  Used for the trained model under evaluation
// and for the "random_init" anchor (which is just a freshly-built ActorCritic
// with the original orthogonal initialisation, never trained).
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

// UniformPolicy — sample one of the legal actions with equal probability.
//
// No network involved; this is the principled "random" baseline (every
// trained model should beat it by a wide margin).
class UniformPolicy : public IPolicy {
public:
    [[nodiscard]] std::string name() const override { return "uniform"; }

    torch::Tensor select_actions(const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask) override {
        (void)obs;
        // multinomial samples proportional to weights; passing the {0,1} mask
        // gives an exactly-uniform distribution over legal slots.
        return legal_mask.multinomial(/*num_samples=*/1, /*replacement=*/true)
                        .squeeze(-1)
                        .to(torch::kInt64);
    }
};

// Helper: walk the legal mask in priority order and return the first legal
// slot.  All rule-based policies use this to build a *guaranteed-legal*
// fallback — the env doesn't always offer Fold (e.g., unraised flop where
// you can check for free), so a hard-coded "fall back to fold" wedge would
// crash with "agent selected an illegal action".
inline int64_t first_legal(const torch::TensorAccessor<float, 1>& mask,
                           int64_t A,
                           std::initializer_list<int64_t> priority) {
    for (int64_t k : priority) {
        if (k >= 0 && k < A && mask[k] > 0.5f) return k;
    }
    // Last resort: scan everything. The env always has at least one legal
    // action when it's the agent's turn, so this is guaranteed to succeed.
    for (int64_t k = 0; k < A; ++k) {
        if (mask[k] > 0.5f) return k;
    }
    return 0;
}

// AlwaysCallPolicy — prefer Check/Call (1); fall back to fold (0) or any
// legal action.
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

// AlwaysRaisePolicy — prefer the smallest legal raise, then call/check, then
// fold; finally any legal action as a last-resort safety net.
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
            // Find the smallest legal raise (action 2..A-1).
            int64_t smallest_raise = -1;
            for (int64_t k = 2; k < A; ++k) {
                if (m[i][k] > 0.5f) { smallest_raise = k; break; }
            }
            a[i] = first_legal(m[i], A, {smallest_raise, 1, 0});
        }
        return out;
    }
};

// PairAllInPolicy — shove with any pocket pair, fold otherwise.
//
// Reads the acting player's hole-card one-hot from obs[0:52]; two cards share
// rank iff their card_ids divide to the same value (card_id = rank*4 + suit).
// If the env has been built with a non-52-card deck (short deck, no-deuces),
// the one-hot still occupies the same 52-slot space — non-existent cards are
// simply never set.
//
// "All-in with a pair" prefers the dedicated all-in slot (index A-1 when
// `include_all_in_slot` is on).  When that slot isn't legal we fall back to
// the largest legal raise, then to Check/Call, and finally Fold — so the
// policy is always able to act regardless of the betting structure.
//
// "Fold when not paired" is the intent; if the env doesn't expose Fold (no
// bet faced — e.g., BB preflop unraised) we fall through to Check, which is
// the cheapest way to stay in for free.
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
                // Largest legal raise slot (which is the all-in slot when it
                // exposes a distinct amount; otherwise the largest pot-fraction).
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
