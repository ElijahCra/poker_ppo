#include "network.h"
#include <torch/torch.h>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::ActorCriticImpl(int obs_dim, int action_count,
                                 int hidden_dim, int num_layers)
{
    // Build shared trunk
    torch::nn::Sequential trunk;
    int in_dim = obs_dim;
    for (int i = 0; i < num_layers; ++i) {
        trunk->push_back(torch::nn::Linear(in_dim, hidden_dim));
        trunk->push_back(torch::nn::Tanh());
        in_dim = hidden_dim;
    }
    trunk_ = register_module("trunk", trunk);

    // Actor and critic heads
    actor_head_  = register_module("actor_head",
                                   torch::nn::Linear(hidden_dim, action_count));
    critic_head_ = register_module("critic_head",
                                   torch::nn::Linear(hidden_dim, 1));

    // Orthogonal init with gain = sqrt(2) for hidden layers
    for (auto& m : trunk_->modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
            torch::nn::init::constant_(lin->bias, 0.0);
        }
    }
    torch::nn::init::orthogonal_(actor_head_->weight, 0.01);
    torch::nn::init::constant_(actor_head_->bias, 0.0);
    torch::nn::init::orthogonal_(critic_head_->weight, 1.0);
    torch::nn::init::constant_(critic_head_->bias, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
std::pair<torch::Tensor, torch::Tensor>
ActorCriticImpl::forward(torch::Tensor obs) {
    auto features = trunk_->forward(obs);
    auto logits   = actor_head_->forward(features);
    auto value    = critic_head_->forward(features).squeeze(-1);
    return {logits, value};
}

// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor
ActorCriticImpl::apply_mask(torch::Tensor logits, torch::Tensor mask) {
    // mask: 1 = legal, 0 = illegal
    return logits + (1.0f - mask) * (-1e8f);
}

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::ActionResult
ActorCriticImpl::get_action(torch::Tensor obs, torch::Tensor legal_mask) {
    auto [logits, value] = forward(obs);
    auto masked = apply_mask(logits, legal_mask);

    auto dist    = torch::softmax(masked, /*dim=*/-1);
    auto action  = dist.multinomial(/*num_samples=*/1, /*replacement=*/true)
                       .squeeze(-1);

    auto log_dist = torch::log_softmax(masked, -1);
    auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy  = -(dist * log_dist).sum(-1);

    return {action, log_prob, value, entropy};
}

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::EvalResult
ActorCriticImpl::evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                          const torch::Tensor &action) {
    auto [logits, value] = forward(std::move(obs));
    const auto masked = apply_mask(logits, std::move(legal_mask));

    const auto dist     = torch::softmax(masked, -1);
    const auto log_dist = torch::log_softmax(masked, -1);
    const auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    const auto entropy  = -(dist * log_dist).sum(-1);

    return {log_prob, value, entropy};
}

} // namespace poker_ppo
