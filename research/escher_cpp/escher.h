// Neural ESCHER (Deep-CFR-style) in libtorch C++ — port of
// research/escher/escher_nn.py, validated on Kuhn / Leduc (exact recursive BR
// as judge). Mirrors the validated Python exactly:
//   - chance/deal-probability weighting in the regret traversal,
//   - PRIVILEGED value features that always include the public card,
//   - fresh-init regret net each iter (stable averaging),
//   - value: replay buffer + frozen-target fitted-value-iteration + output
//     scaling + on-σ-mixed sampling + K sweeps/iter.
#pragma once

#include <torch/torch.h>

#include <functional>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "games.h"
#include "solvers.h"

namespace escher {

// ─── tiny MLP ───────────────────────────────────────────────────────────────
struct MLPImpl : torch::nn::Module {
    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    MLPImpl(int d_in, int d_out, int hidden) {
        l1 = register_module("l1", torch::nn::Linear(d_in, hidden));
        l2 = register_module("l2", torch::nn::Linear(hidden, hidden));
        l3 = register_module("l3", torch::nn::Linear(hidden, d_out));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(l1(x));
        x = torch::relu(l2(x));
        return l3(x);
    }
};
TORCH_MODULE(MLP);

// ─── feature encoding ───────────────────────────────────────────────────────
class Featurizer {
public:
    Featurizer(const Game& g, int n_ranks);
    std::vector<float> infoset(const std::string& h, const Cards& c, int player) const;
    std::vector<float> full(const std::string& h, const Cards& c) const;
    int inf_dim() const { return inf_dim_; }
    int full_dim() const { return full_dim_; }

private:
    int public_idx(const std::string& h, const Cards& c) const {
        if (!has_public_) return 0;
        return (h.find('/') != std::string::npos) ? 1 + c[2] : 0;
    }
    const Game& g_;
    int n_ranks_;
    bool has_public_;
    std::unordered_map<std::string, int> hidx_;
    int n_hist_, pub_dim_, inf_dim_, full_dim_, extra_dim_;
};

// ─── reservoir buffer ───────────────────────────────────────────────────────
struct Sample {
    std::vector<float> feat;
    std::vector<float> target;  // regret vector or strat-prob vector
    float w;
};

class Reservoir {
public:
    Reservoir(size_t cap, uint32_t seed) : cap_(cap), rng_(seed) {}
    void add(Sample s) {
        ++n_;
        if (data_.size() < cap_) { data_.push_back(std::move(s)); return; }
        std::uniform_int_distribution<long> d(0, n_ - 1);
        long j = d(rng_);
        if (j < static_cast<long>(cap_)) data_[j] = std::move(s);
    }
    std::vector<Sample>& data() { return data_; }
    size_t size() const { return data_.size(); }

private:
    size_t cap_;
    std::vector<Sample> data_;
    long n_ = 0;
    std::mt19937 rng_;
};

// state buffer for value learning: (history, cards) visited under sampling
struct StateSample { std::string h; Cards c; };
class StateReservoir {
public:
    StateReservoir(size_t cap, uint32_t seed) : cap_(cap), rng_(seed) {}
    void add(StateSample s) {
        ++n_;
        if (data_.size() < cap_) { data_.push_back(std::move(s)); return; }
        std::uniform_int_distribution<long> d(0, n_ - 1);
        long j = d(rng_);
        if (j < static_cast<long>(cap_)) data_[j] = std::move(s);
    }
    std::vector<StateSample>& data() { return data_; }
    size_t size() const { return data_.size(); }

private:
    size_t cap_;
    std::vector<StateSample> data_;
    long n_ = 0;
    std::mt19937 rng_;
};

// ─── Neural ESCHER ──────────────────────────────────────────────────────────
class NeuralESCHER {
public:
    enum class Mode { Exact, Net };
    struct Config {
        Mode mode = Mode::Exact;
        int hidden = 64;
        int val_hidden = 256;
        int reg_steps = 400;
        int val_sweeps = 4;
        int seed = 0;
    };

    NeuralESCHER(const Game& g, int n_ranks, Config cfg);

    // run `iters` CFR iterations; print exploitability every `eval_every`.
    void run(int iters, int eval_every);

private:
    std::vector<double> sigma(const std::string& h, const Cards& c,
                              int player, const std::vector<int>& legal);
    double exact_value(const std::string& h, const Cards& c);
    double net_value(const std::string& h, const Cards& c);
    double value_of(const std::string& h, const Cards& c) {
        return cfg_.mode == Mode::Net ? net_value(h, c) : exact_value(h, c);
    }
    void collect_regret(long t);
    void fit_regret();
    void fit_avg();
    void train_value();
    Strategy tabular_average();
    std::pair<double, double> value_diag();
    int choose_sigma(const std::vector<int>& legal, const std::vector<double>& sig);
    Cards sample_deal();
    double compute_v_scale();

    torch::Tensor feat_tensor(const std::vector<float>& f) const {
        return torch::from_blob(const_cast<float*>(f.data()),
                                {1, static_cast<long>(f.size())},
                                torch::kFloat).clone();
    }

    const Game& g_;
    Featurizer feat_;
    Config cfg_;
    int A_;
    long t_ = 0;
    std::mt19937 rng_;

    MLP regret_net_{nullptr}, avg_net_{nullptr}, value_net_{nullptr};
    std::shared_ptr<torch::optim::Adam> value_opt_;
    Reservoir reg_buf_, strat_buf_;
    StateReservoir val_buf_;
    double v_scale_ = 1.0;
    double v_expl_ = 0.6;

    // exact tabular linear-average accumulator (the deployed strategy)
    std::unordered_map<std::string, std::vector<double>> tab_ss_;
    std::unordered_map<std::string, double> tab_w_;
    std::unordered_map<std::string, std::vector<int>> tab_legal_;
};

}  // namespace escher
