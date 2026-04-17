#include "network.h"
#include <torch/torch.h>

namespace poker_ppo {

// ═════════════════════════════════════════════════════════════════════════════
// Helper: build an MLP trunk with orthogonal init
// ═════════════════════════════════════════════════════════════════════════════

static torch::nn::Sequential build_trunk(int in_dim, int hidden_dim,
                                         int num_layers) {
    torch::nn::Sequential trunk;
    int dim = in_dim;
    for (int i = 0; i < num_layers; ++i) {
        trunk->push_back(torch::nn::Linear(dim, hidden_dim));
        trunk->push_back(torch::nn::Tanh());
        dim = hidden_dim;
    }

    // Orthogonal init with gain = sqrt(2) for hidden layers
    for (auto& m : trunk->modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
            torch::nn::init::constant_(lin->bias, 0.0);
        }
    }
    return trunk;
}

// ═════════════════════════════════════════════════════════════════════════════
// Actor
// ═════════════════════════════════════════════════════════════════════════════

ActorImpl::ActorImpl(int obs_dim, int action_count,
                     int hidden_dim, int num_layers) {
    trunk_ = register_module("trunk", build_trunk(obs_dim, hidden_dim, num_layers));
    head_  = register_module("head", torch::nn::Linear(hidden_dim, action_count));

    // Small init for policy head — keeps initial policy near-uniform
    torch::nn::init::orthogonal_(head_->weight, 0.01);
    torch::nn::init::constant_(head_->bias, 0.0);
}

torch::Tensor ActorImpl::forward(torch::Tensor obs) {
    return head_->forward(trunk_->forward(obs));
}

torch::Tensor ActorImpl::apply_mask(torch::Tensor logits, torch::Tensor mask) {
    // mask: 1 = legal, 0 = illegal
    return logits + (1.0f - mask) * (-1e8f);
}

ActorImpl::ActionResult
ActorImpl::get_action(torch::Tensor obs, torch::Tensor legal_mask) {
    auto logits = forward(obs);
    auto masked = apply_mask(logits, legal_mask);

    auto probs    = torch::softmax(masked, /*dim=*/-1);
    auto action   = probs.multinomial(/*num_samples=*/1, /*replacement=*/true)
                         .squeeze(-1);
    auto log_dist = torch::log_softmax(masked, -1);
    auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy  = -(probs * log_dist).sum(-1);

    return {action, log_prob, entropy};
}

ActorImpl::EvalResult
ActorImpl::evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                    torch::Tensor action) {
    auto logits = forward(obs);
    auto masked = apply_mask(logits, legal_mask);

    auto probs    = torch::softmax(masked, -1);
    auto log_dist = torch::log_softmax(masked, -1);
    auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy  = -(probs * log_dist).sum(-1);

    return {log_prob, entropy};
}

// ═════════════════════════════════════════════════════════════════════════════
// Critic
// ═════════════════════════════════════════════════════════════════════════════

CriticImpl::CriticImpl(int obs_dim, int hidden_dim, int num_layers) {
    trunk_ = register_module("trunk", build_trunk(obs_dim, hidden_dim, num_layers));
    head_  = register_module("head", torch::nn::Linear(hidden_dim, 1));

    // Critic head gets gain=1.0
    torch::nn::init::orthogonal_(head_->weight, 1.0);
    torch::nn::init::constant_(head_->bias, 0.0);
}

torch::Tensor CriticImpl::forward(torch::Tensor obs) {
    return head_->forward(trunk_->forward(obs)).squeeze(-1);
}

} // namespace poker_ppo
