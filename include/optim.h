#pragma once
//
// Multi-tensor ("foreach") Adam + grad-norm clipping.
//
// libtorch's C++-frontend torch::optim::Adam loops over parameters and
// launches ~7 tiny CUDA kernels per tensor per step; ~50 parameter tensors
// × 16 minibatch steps per update is thousands of microsecond kernels —
// pure launch overhead on a model this small. The at::_foreach_* horizontal
// ops do the same math in a handful of fused launches regardless of
// parameter count. Same applies to torch::nn::utils::clip_grad_norm_,
// which computes one norm kernel per gradient tensor.
//
// The math matches torch::optim::Adam bit-for-bit modulo kernel reduction
// order — see Game/tests/optim_test.cpp for the equivalence test.
//

#include <torch/torch.h>

#include <vector>

namespace poker_ppo {

class ForeachAdam {
public:
    // Hyperparameter defaults mirror torch::optim::AdamOptions.
    explicit ForeachAdam(std::vector<torch::Tensor> params,
                         double lr,
                         double beta1 = 0.9,
                         double beta2 = 0.999,
                         double eps   = 1e-8);

    void   set_lr(double lr) noexcept { lr_ = lr; }
    double lr() const noexcept        { return lr_; }

    [[nodiscard]] const std::vector<torch::Tensor>& params() const noexcept {
        return params_;
    }

    // set_to_none semantics: backward() then *assigns* fresh gradients
    // instead of accumulating into zeroed ones, so the zeroing kernels
    // disappear entirely.
    void zero_grad();

    void step();

    // Step from externally-held gradient tensors (same order as params()),
    // bypassing .grad() — used by the CUDA-graphed update, whose backward
    // writes into static graph-pool tensors instead of param.grad().
    void step(const std::vector<torch::Tensor>& grads);

private:
    std::vector<torch::Tensor> params_;
    std::vector<torch::Tensor> exp_avg_;
    std::vector<torch::Tensor> exp_avg_sq_;
    double  lr_, beta1_, beta2_, eps_;
    int64_t step_count_ = 0;
};

// clip_grad_norm_ (norm_type=2) without the per-tensor norm kernels and
// without a host sync: the clip coefficient stays on device and is applied
// unconditionally after clamping to 1 (x *= 1.0 is exact, so the no-clip
// case is unchanged). Returns the pre-clip total norm (device scalar).
torch::Tensor foreach_clip_grad_norm(const std::vector<torch::Tensor>& params,
                                     double max_norm);

// Same, but on gradient tensors directly (CUDA-graphed update path).
torch::Tensor foreach_clip_grads(const std::vector<torch::Tensor>& grads,
                                 double max_norm);

}  // namespace poker_ppo
