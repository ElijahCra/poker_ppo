#include "optim.h"

#include <cmath>

namespace poker_ppo {

ForeachAdam::ForeachAdam(std::vector<torch::Tensor> params,
                         double lr, double beta1, double beta2, double eps)
    : params_(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
{
    torch::NoGradGuard ng;
    exp_avg_.reserve(params_.size());
    exp_avg_sq_.reserve(params_.size());
    for (const auto& p : params_) {
        exp_avg_.push_back(torch::zeros_like(p));
        exp_avg_sq_.push_back(torch::zeros_like(p));
    }
}

void ForeachAdam::load_state(const std::vector<torch::Tensor>& m,
                             const std::vector<torch::Tensor>& v,
                             int64_t step) {
    TORCH_CHECK(m.size() == exp_avg_.size() && v.size() == exp_avg_sq_.size(),
                "ForeachAdam::load_state: moment count mismatch");
    torch::NoGradGuard ng;
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        exp_avg_[i].copy_(m[i]);       // copy_ handles device/dtype
        exp_avg_sq_[i].copy_(v[i]);
    }
    step_count_ = step;
}

void ForeachAdam::zero_grad() {
    for (auto& p : params_) p.mutable_grad().reset();
}

void ForeachAdam::step() {
    std::vector<torch::Tensor> grads;
    grads.reserve(params_.size());
    for (const auto& p : params_) {
        TORCH_CHECK(p.grad().defined(),
                    "ForeachAdam::step: parameter without gradient");
        grads.push_back(p.grad());
    }
    step(grads);
}

void ForeachAdam::step(const std::vector<torch::Tensor>& grads) {
    TORCH_CHECK(grads.size() == params_.size(),
                "ForeachAdam::step: grads/params size mismatch");
    torch::NoGradGuard ng;
    ++step_count_;

    // Same update as torch::optim::Adam (no weight decay, no amsgrad):
    //   m ← β₁m + (1−β₁)g
    //   v ← β₂v + (1−β₂)g²
    //   p ← p − lr/(1−β₁ᵗ) · m / (√(v/(1−β₂ᵗ)) + ε)
    at::_foreach_mul_(exp_avg_, beta1_);
    at::_foreach_add_(exp_avg_, grads, 1.0 - beta1_);
    at::_foreach_mul_(exp_avg_sq_, beta2_);
    at::_foreach_addcmul_(exp_avg_sq_, grads, grads, 1.0 - beta2_);

    const double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
    const double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

    auto denom = at::_foreach_sqrt(exp_avg_sq_);
    at::_foreach_div_(denom, std::sqrt(bias_correction2));
    at::_foreach_add_(denom, eps_);
    at::_foreach_addcdiv_(params_, exp_avg_, denom, -lr_ / bias_correction1);
}

torch::Tensor foreach_clip_grads(const std::vector<torch::Tensor>& grads,
                                 double max_norm) {
    torch::NoGradGuard ng;
    if (grads.empty()) return torch::zeros({});

    auto total_norm = torch::norm(torch::stack(at::_foreach_norm(grads, 2)), 2);
    // 1e-6 matches torch::nn::utils::clip_grad_norm_.
    auto clip_coef  = (max_norm / (total_norm + 1e-6)).clamp_max(1.0);
    at::_foreach_mul_(grads, clip_coef);
    return total_norm;
}

torch::Tensor foreach_clip_grad_norm(const std::vector<torch::Tensor>& params,
                                     double max_norm) {
    std::vector<torch::Tensor> grads;
    grads.reserve(params.size());
    for (const auto& p : params) {
        if (p.grad().defined()) grads.push_back(p.grad());
    }
    return foreach_clip_grads(grads, max_norm);
}

}  // namespace poker_ppo
