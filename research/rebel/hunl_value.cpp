#include "hunl_value.h"

#include "hunl_gpu.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "config.h"

namespace rebel_hunl {

namespace {
int street_of(int nb) {
    switch (nb) {
    case 0: return 0;
    case 3: return 1;
    case 4: return 2;
    default: return 3;
    }
}

// dataset file header — kdim/nout guard against silently loading rows
// featurized by an older HunlFeaturizer
constexpr uint32_t kDataMagic   = 0x444C4252u;   // "RBLD"
constexpr uint32_t kDataVersion = 1;

bool check_data_header(const std::string& path, int n_out) {
    std::ifstream f(path, std::ios::binary);
    uint32_t hdr[4];
    if (!f.read(reinterpret_cast<char*>(hdr), sizeof hdr)) return false;
    return hdr[0] == kDataMagic && hdr[1] == kDataVersion &&
           hdr[2] == static_cast<uint32_t>(HunlFeaturizer::kDim) &&
           hdr[3] == static_cast<uint32_t>(n_out);
}
}  // namespace

namespace {
// eq[i] = (win − lose mass of combo i vs opp) / compatible mass ∈ [−1,1].
// Scale-invariant in opp's normalization — consistent between training
// rows (normalized ranges) and solver queries (solver-scaled beliefs).
void equity_vs_range(const RiverEval& ev, const std::vector<double>& opp,
                     std::vector<double>& eq) {
    std::vector<double> num, den;
    ev.cfv(opp, 1.0, num);
    compat_mass(opp, ev.valid(), den);
    eq.assign(kCombos, 0.0);
    for (int i = 0; i < kCombos; ++i)
        if (den[i] > 0.0) eq[i] = num[i] / den[i];
}
}  // namespace

std::vector<float> HunlFeaturizer::features(int street, const uint8_t* board,
                                            int nb, double pot, double stack,
                                            const HunlPBS& beta) {
    static const std::vector<double> kZero(kCombos, 0.0);
    if (nb == 5) {
        RiverEval ev(board);
        std::vector<double> pct, eq0, eq1;
        ev.percentile(pct);
        equity_vs_range(ev, beta.r1, eq0);
        equity_vs_range(ev, beta.r0, eq1);
        return features(street, board, nb, pot, stack, beta, pct, eq0, eq1);
    }
    if (nb == 4) {
        // Turn-root rows (Phase C): expectation of the river strength
        // blocks over runout cards the combo survives — DeepStack's E[HS]
        // one street early. ~48 river evals per call; turn-root targets
        // are built once per solve, so the cost is training-time only.
        const auto& ct = ComboTable::get();
        std::vector<double> pct(kCombos, 0.0), eq0(kCombos, 0.0),
            eq1(kCombos, 0.0), n(kCombos, 0.0);
        std::array<uint8_t, 5> b5{};
        for (int b = 0; b < 4; ++b) b5[b] = board[b];
        std::vector<double> pc, e0, e1, opp;
        for (int c = 0; c < kCards; ++c) {
            bool dead = false;
            for (int b = 0; b < 4; ++b)
                if (board[b] == c) dead = true;
            if (dead) continue;
            b5[4] = static_cast<uint8_t>(c);
            RiverEval ev(b5.data());
            ev.percentile(pc);
            auto masked = [&](const std::vector<double>& r) {
                opp = r;
                for (int i = 0; i < kCombos; ++i)
                    if (ct.cards[i][0] == c || ct.cards[i][1] == c)
                        opp[i] = 0.0;
                return &opp;
            };
            equity_vs_range(ev, *masked(beta.r1), e0);
            std::vector<double> e0copy = e0;
            equity_vs_range(ev, *masked(beta.r0), e1);
            for (int x = 0; x < kCombos; ++x) {
                if (ct.cards[x][0] == c || ct.cards[x][1] == c) continue;
                if (!ev.valid()[x]) continue;
                pct[x] += pc[x];
                eq0[x] += e0copy[x];
                eq1[x] += e1[x];
                n[x] += 1.0;
            }
        }
        for (int x = 0; x < kCombos; ++x)
            if (n[x] > 0.0) {
                pct[x] /= n[x];
                eq0[x] /= n[x];
                eq1[x] /= n[x];
            }
        return features(street, board, nb, pot, stack, beta, pct, eq0, eq1);
    }
    if (nb == 3) {
        // Flop-root rows / preflop-solve leaf queries: strength blocks as
        // an expectation over a FIXED per-board runout subset (32 of the
        // 1176 turn+river completions, seeded by a board hash) — the
        // determinism is the self-consistency requirement: training rows
        // and play-time queries at the same flop compute identical
        // features. Exact two-street expectation is ~35× the cost for a
        // sub-noise refinement.
        const auto& ct = ComboTable::get();
        std::vector<double> pct(kCombos, 0.0), eq0(kCombos, 0.0),
            eq1(kCombos, 0.0), n(kCombos, 0.0);
        uint64_t key = 1469598103934665603ull;
        for (int b = 0; b < 3; ++b)
            key = (key ^ board[b]) * 1099511628211ull;
        std::mt19937_64 rr(key);
        std::vector<uint8_t> rem;
        for (int c = 0; c < kCards; ++c) {
            bool dead = false;
            for (int b = 0; b < 3; ++b)
                if (board[b] == c) dead = true;
            if (!dead) rem.push_back(static_cast<uint8_t>(c));
        }
        std::array<uint8_t, 5> b5{};
        for (int b = 0; b < 3; ++b) b5[b] = board[b];
        std::vector<double> pc, e0, e1, opp;
        for (int s2 = 0; s2 < 32; ++s2) {
            const size_t i1 = rr() % rem.size();
            size_t i2 = rr() % (rem.size() - 1);
            if (i2 >= i1) ++i2;
            b5[3] = rem[i1];
            b5[4] = rem[i2];
            RiverEval ev(b5.data());
            ev.percentile(pc);
            auto masked = [&](const std::vector<double>& r) {
                opp = r;
                for (int i = 0; i < kCombos; ++i)
                    if (ct.cards[i][0] == b5[3] || ct.cards[i][1] == b5[3] ||
                        ct.cards[i][0] == b5[4] || ct.cards[i][1] == b5[4])
                        opp[i] = 0.0;
                return &opp;
            };
            equity_vs_range(ev, *masked(beta.r1), e0);
            std::vector<double> e0copy = e0;
            equity_vs_range(ev, *masked(beta.r0), e1);
            for (int x = 0; x < kCombos; ++x) {
                if (!ev.valid()[x]) continue;
                pct[x] += pc[x];
                eq0[x] += e0copy[x];
                eq1[x] += e1[x];
                n[x] += 1.0;
            }
        }
        for (int x = 0; x < kCombos; ++x)
            if (n[x] > 0.0) {
                pct[x] /= n[x];
                eq0[x] /= n[x];
                eq1[x] /= n[x];
            }
        return features(street, board, nb, pot, stack, beta, pct, eq0, eq1);
    }
    // preflop root: strength blocks pending (train_preflop stage)
    return features(street, board, nb, pot, stack, beta, kZero, kZero,
                    kZero);
}

std::vector<float> HunlFeaturizer::features(int street, const uint8_t* board,
                                            int nb, double pot, double stack,
                                            const HunlPBS& beta,
                                            const std::vector<double>& pct,
                                            const std::vector<double>& eq0,
                                            const std::vector<double>& eq1) {
    std::vector<float> f(kDim, 0.0f);
    f[street] = 1.0f;
    for (int i = 0; i < nb; ++i) f[4 + board[i]] = 1.0f;
    f[4 + kCards] = static_cast<float>(pot / stack);
    const int off = 4 + kCards + 1;
    for (int i = 0; i < kCombos; ++i) {
        f[off + i] = static_cast<float>(beta.r0[i]);
        f[off + kCombos + i] = static_cast<float>(beta.r1[i]);
        f[off + 2 * kCombos + i] = static_cast<float>(pct[i]);
        f[off + 3 * kCombos + i] = static_cast<float>(eq0[i]);
        f[off + 4 * kCombos + i] = static_cast<float>(eq1[i]);
    }
    return f;
}

HunlValueNetImpl::HunlValueNetImpl(int hidden, int layers, bool gelu_ln,
                                   bool zero_sum_on) {
    gelu = gelu_ln;
    zero_sum = zero_sum_on;
    zs_M = torch::zeros({kCombos, kCards});
    {
        auto acc = zs_M.accessor<float, 2>();
        const auto& ct = ComboTable::get();
        for (int i = 0; i < kCombos; ++i) {
            acc[i][ct.cards[i][0]] = 1.0f;
            acc[i][ct.cards[i][1]] = 1.0f;
        }
    }
    l1 = register_module("l1", torch::nn::Linear(HunlFeaturizer::kDim, hidden));
    if (gelu)
        lns.push_back(register_module(
            "ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                       {hidden}))));
    for (int i = 1; i < layers; ++i) {
        const std::string nm = i == 1 ? "l2" : "l2_" + std::to_string(i);
        mids.push_back(
            register_module(nm, torch::nn::Linear(hidden, hidden)));
        if (gelu)
            lns.push_back(register_module(
                "ln" + std::to_string(i + 1),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden}))));
    }
    l3 = register_module("l3", torch::nn::Linear(hidden, 2 * kCombos));
}

void HunlValueNetImpl::to(torch::Device device, bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    if (zs_M.defined()) zs_M = zs_M.to(device);
}

torch::Tensor HunlValueNetImpl::forward(torch::Tensor x) {
    torch::Tensor h = l1(x);
    h = gelu ? torch::gelu(lns[0](h)) : torch::relu(h);
    for (size_t k = 0; k < mids.size(); ++k) {
        h = mids[k](h);
        h = gelu ? torch::gelu(lns[k + 1](h)) : torch::relu(h);
    }
    auto y = l3(h);
    if (!zero_sum) return y;
    // DeepStack's outer step: estimate both players' game values from the
    // input ranges, then subtract half the (should-be-zero) sum from every
    // value entry. Compatible-mass weights come from inclusion-exclusion:
    // m0_i = S1 - A1[a_i] - A1[b_i] + r1_i with A1 = per-card mass of r1.
    // Scale-invariant in the range normalization (s and Z scale together).
    constexpr int off = 4 + kCards + 1;
    auto r0 = x.slice(1, off, off + kCombos);
    auto r1 = x.slice(1, off + kCombos, off + 2 * kCombos);
    auto A0 = torch::matmul(r0, zs_M);                       // [B, 52]
    auto A1 = torch::matmul(r1, zs_M);
    auto m0 = r1.sum(1, true) - torch::matmul(A1, zs_M.t()) + r1;
    auto m1 = r0.sum(1, true) - torch::matmul(A0, zs_M.t()) + r0;
    auto w0 = r0 * m0, w1 = r1 * m1;                         // [B, kCombos]
    auto Z = w0.sum(1, true);                                // = w1 sum
    auto s = (w0 * y.slice(1, 0, kCombos)).sum(1, true) +
             (w1 * y.slice(1, kCombos, 2 * kCombos)).sum(1, true);
    return y - s / (2.0 * Z.clamp_min(1e-9));
}

void HunlNetOracle::value(poker_ppo::PokerEnvironment& env,
                          const uint8_t* board, int nb, const HunlPBS& beta,
                          std::array<std::vector<double>, 2>& out) {
    std::vector<uint8_t> cards = {board[nb - 1]};
    std::vector<HunlPBS> betas = {beta};
    std::vector<std::array<std::vector<double>, 2>> outs;
    value_batch(env, board, nb - 1, cards, betas, outs);
    out = std::move(outs[0]);
}

void HunlNetOracle::value_batch(
    poker_ppo::PokerEnvironment& env, const uint8_t* base_board, int nb_base,
    const std::vector<uint8_t>& cards, const std::vector<HunlPBS>& betas,
    std::vector<std::array<std::vector<double>, 2>>& outs) {
    const long N = static_cast<long>(cards.size());
    outs.resize(cards.size());
    if (N == 0) return;
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    const double pot = 2.0 * contrib;
    torch::NoGradGuard ng;
    auto X = torch::empty({N, HunlFeaturizer::kDim}, torch::kFloat);
    std::array<uint8_t, 5> b{};
    for (int i = 0; i < nb_base; ++i) b[i] = base_board[i];
    // new base board (new solve) → drop the per-card strength contexts
    if (ctx_nb_ != nb_base ||
        !std::equal(b.begin(), b.begin() + nb_base, ctx_board_.begin())) {
        for (auto& c : ctx_) c = CardCtx{};
        ctx_board_ = b;
        ctx_nb_ = nb_base;
    }
    const auto& ct = ComboTable::get();
    std::vector<double> eq0, eq1;
    for (long r = 0; r < N; ++r) {
        const uint8_t card = cards[static_cast<size_t>(r)];
        b[nb_base] = card;
        std::vector<float> f;
        if (nb_base + 1 == 5) {
            CardCtx& cc = ctx_[card];
            if (!cc.ev) {
                cc.ev = std::make_unique<RiverEval>(b.data());
                cc.ev->percentile(cc.pct);
            }
            const HunlPBS& beta = betas[static_cast<size_t>(r)];
            equity_vs_range(*cc.ev, beta.r1, eq0);
            equity_vs_range(*cc.ev, beta.r0, eq1);
            f = HunlFeaturizer::features(street_of(5), b.data(), 5, pot,
                                         stack_, beta, cc.pct, eq0, eq1);
        } else if (nb_base + 1 == 4) {
            // turn-root query (flop solve leaf): E-over-river strength
            // blocks from the cached per-runout evaluators — building
            // rank/sort contexts per query would dominate the solve
            CardCtx& cc = ctx_[card];
            if (cc.pct.empty()) {
                cc.pct.assign(kCombos, 0.0);
                std::vector<double> nrun(kCombos, 0.0), pc;
                std::array<uint8_t, 5> b5{};
                for (int i = 0; i < 4; ++i) b5[i] = b[i];
                for (int c = 0; c < kCards; ++c) {
                    bool dead = false;
                    for (int i = 0; i < 4; ++i)
                        if (b[i] == c) dead = true;
                    if (dead) continue;
                    b5[4] = static_cast<uint8_t>(c);
                    cc.runout[c] = std::make_unique<RiverEval>(b5.data());
                    cc.runout[c]->percentile(pc);
                    for (int x = 0; x < kCombos; ++x) {
                        if (!cc.runout[c]->valid()[x]) continue;
                        cc.pct[x] += pc[x];
                        nrun[x] += 1.0;
                    }
                }
                for (int x = 0; x < kCombos; ++x)
                    if (nrun[x] > 0.0) cc.pct[x] /= nrun[x];
            }
            const HunlPBS& beta = betas[static_cast<size_t>(r)];
            eq0.assign(kCombos, 0.0);
            eq1.assign(kCombos, 0.0);
            std::vector<double> nrun(kCombos, 0.0), e, opp;
            for (int c = 0; c < kCards; ++c) {
                if (!cc.runout[c]) continue;
                const RiverEval& ev = *cc.runout[c];
                for (int side = 0; side < 2; ++side) {
                    opp = side == 0 ? beta.r1 : beta.r0;
                    for (int i = 0; i < kCombos; ++i)
                        if (ct.cards[i][0] == c || ct.cards[i][1] == c)
                            opp[i] = 0.0;
                    equity_vs_range(ev, opp, e);
                    auto& acc = side == 0 ? eq0 : eq1;
                    for (int x = 0; x < kCombos; ++x) {
                        if (!ev.valid()[x] || ct.cards[x][0] == c ||
                            ct.cards[x][1] == c)
                            continue;
                        acc[x] += e[x];
                        if (side == 0) nrun[x] += 1.0;
                    }
                }
            }
            for (int x = 0; x < kCombos; ++x)
                if (nrun[x] > 0.0) {
                    eq0[x] /= nrun[x];
                    eq1[x] /= nrun[x];
                }
            f = HunlFeaturizer::features(street_of(4), b.data(), 4, pot,
                                         stack_, beta, cc.pct, eq0, eq1);
        } else {
            f = HunlFeaturizer::features(street_of(nb_base + 1), b.data(),
                                         nb_base + 1, pot, stack_,
                                         betas[static_cast<size_t>(r)]);
        }
        std::copy(f.begin(), f.end(),
                  X.data_ptr<float>() +
                      static_cast<size_t>(r) * HunlFeaturizer::kDim);
    }
    // ONE forward for the whole leaf (all candidate cards) on the device.
    // Net predicts values in POT units (paper normalization).
    auto y = net_->forward(X.to(device_)).to(torch::kCPU).contiguous();
    auto acc = y.accessor<float, 2>();
    for (long r = 0; r < N; ++r)
        for (int p = 0; p < 2; ++p) {
            auto& o = outs[static_cast<size_t>(r)][p];
            o.assign(kCombos, 0.0);
            for (int i = 0; i < kCombos; ++i)
                o[i] = static_cast<double>(acc[r][p * kCombos + i]) * pot;
        }
}

int convert_dataset(const std::string& in, const std::string& out) {
    constexpr int kOldDim = 4 + kCards + 1 + 2 * kCombos;   // 2709
    const int n_out = 2 * kCombos;
    const double stack =
        static_cast<double>(poker_ppo::kPokerConfig.game.initial_stack);
    std::ifstream fi(in, std::ios::binary);
    if (!fi) {
        std::fprintf(stderr, "convert: cannot open %s\n", in.c_str());
        return 1;
    }
    uint32_t hdr[4];
    if (!fi.read(reinterpret_cast<char*>(hdr), sizeof hdr) ||
        hdr[0] != kDataMagic || hdr[1] != kDataVersion ||
        hdr[2] != static_cast<uint32_t>(kOldDim) ||
        hdr[3] != static_cast<uint32_t>(n_out)) {
        std::fprintf(stderr,
                     "convert: %s is not a kdim=%d dataset (already "
                     "converted?)\n", in.c_str(), kOldDim);
        return 1;
    }
    std::ofstream fo(out, std::ios::binary | std::ios::trunc);
    if (!fo) {
        std::fprintf(stderr, "convert: cannot open %s\n", out.c_str());
        return 1;
    }
    const uint32_t nhdr[4] = {kDataMagic, kDataVersion,
                              static_cast<uint32_t>(HunlFeaturizer::kDim),
                              static_cast<uint32_t>(n_out)};
    fo.write(reinterpret_cast<const char*>(nhdr), sizeof nhdr);

    std::vector<float> feat(kOldDim), target(static_cast<size_t>(n_out));
    std::vector<uint8_t> mask(static_cast<size_t>(n_out));
    long rows = 0;
    while (fi.read(reinterpret_cast<char*>(feat.data()),
                   sizeof(float) * kOldDim) &&
           fi.read(reinterpret_cast<char*>(target.data()),
                   sizeof(float) * static_cast<size_t>(n_out)) &&
           fi.read(reinterpret_cast<char*>(mask.data()), n_out)) {
        int street = 0;
        for (int s = 1; s < 4; ++s)
            if (feat[s] > feat[street]) street = s;
        uint8_t board[5];
        int nb = 0;
        for (int c = 0; c < kCards && nb < 5; ++c)
            if (feat[4 + c] > 0.5f) board[nb++] = static_cast<uint8_t>(c);
        const double pot = static_cast<double>(feat[4 + kCards]) * stack;
        HunlPBS beta;
        beta.r0.resize(kCombos);
        beta.r1.resize(kCombos);
        const int off = 4 + kCards + 1;
        for (int i = 0; i < kCombos; ++i) {
            beta.r0[i] = static_cast<double>(feat[off + i]);
            beta.r1[i] = static_cast<double>(feat[off + kCombos + i]);
        }
        auto nf = HunlFeaturizer::features(street, board, nb, pot, stack,
                                           beta);
        fo.write(reinterpret_cast<const char*>(nf.data()),
                 sizeof(float) * HunlFeaturizer::kDim);
        fo.write(reinterpret_cast<const char*>(target.data()),
                 sizeof(float) * static_cast<size_t>(n_out));
        fo.write(reinterpret_cast<const char*>(mask.data()), n_out);
        if (++rows % 100000 == 0) {
            std::printf("  convert: %ld rows\n", rows);
            std::fflush(stdout);
        }
    }
    std::printf("convert: %s -> %s  %ld rows (kdim %d -> %d)\n", in.c_str(),
                out.c_str(), rows, kOldDim, HunlFeaturizer::kDim);
    return 0;
}

void HunlNetOracle::value_boards(
    poker_ppo::PokerEnvironment& env,
    const std::vector<std::array<uint8_t, 3>>& flops,
    const std::vector<HunlPBS>& betas,
    std::vector<std::array<std::vector<double>, 2>>& outs) {
    const long N = static_cast<long>(flops.size());
    outs.resize(flops.size());
    if (N == 0) return;
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    const double pot = 2.0 * contrib;
    torch::NoGradGuard ng;
    auto X = torch::empty({N, HunlFeaturizer::kDim}, torch::kFloat);
    for (long r = 0; r < N; ++r) {
        auto f = HunlFeaturizer::features(
            1, flops[static_cast<size_t>(r)].data(), 3, pot, stack_,
            betas[static_cast<size_t>(r)]);
        std::copy(f.begin(), f.end(),
                  X.data_ptr<float>() +
                      static_cast<size_t>(r) * HunlFeaturizer::kDim);
    }
    auto y = net_->forward(X.to(device_)).to(torch::kCPU).contiguous();
    auto acc = y.accessor<float, 2>();
    for (long r = 0; r < N; ++r)
        for (int p = 0; p < 2; ++p) {
            auto& o = outs[static_cast<size_t>(r)][p];
            o.assign(kCombos, 0.0);
            for (int i = 0; i < kCombos; ++i)
                o[i] = static_cast<double>(acc[r][p * kCombos + i]) * pot;
        }
}

EndgameTrainer::EndgameTrainer(EndgameConfig cfg)
    : cfg_(std::move(cfg)),
      stack_(static_cast<double>(
          poker_ppo::kPokerConfig.game.initial_stack)),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      rng_(static_cast<unsigned>(cfg_.seed)) {
    if (cfg_.seed == 0) {
        // distinct data streams across resumed runs (seed collisions make a
        // resumed run's "fresh" samples replay the previous run's exactly);
        // pass an explicit seed for reproducibility.
        cfg_.seed = std::random_device{}();
        std::printf("  [seed] %llu\n",
                    static_cast<unsigned long long>(cfg_.seed));
    }
    rng_.seed(static_cast<unsigned>(cfg_.seed));
    torch::manual_seed(cfg_.seed);
    circular_ = cfg_.circular < 0 ? !cfg_.river_only : cfg_.circular != 0;
    net_ = HunlValueNet(cfg_.hidden, cfg_.layers, cfg_.gelu_ln,
                        cfg_.zero_sum);
    if (!cfg_.ckpt.empty() && std::filesystem::exists(cfg_.ckpt)) {
        // torch::load REPLACES parameter tensors, so a foreign-architecture
        // checkpoint loads "successfully" and silently trains the old net.
        // Not recoverable by starting fresh either: the epoch loop would
        // overwrite the good checkpoint with an untrained net. Check shapes.
        std::unordered_map<std::string, std::vector<int64_t>> want;
        for (const auto& p : net_->named_parameters())
            want[p.key()] = p.value().sizes().vec();
        bool ok = true;
        try {
            torch::load(net_, cfg_.ckpt);
            for (const auto& p : net_->named_parameters())
                if (want.at(p.key()) != p.value().sizes().vec()) ok = false;
        } catch (const std::exception&) {
            ok = false;
        }
        if (!ok) {
            std::fprintf(stderr,
                         "  [ckpt] %s does not fit hidden=%d layers=%d — set "
                         "REBEL_CKPT to a fresh path for this architecture\n",
                         cfg_.ckpt.c_str(), cfg_.hidden, cfg_.layers);
            std::exit(1);
        }
        std::printf("  [ckpt] loaded %s\n", cfg_.ckpt.c_str());
    }
    net_->to(device_);
    opt_ = std::make_unique<torch::optim::Adam>(net_->parameters(), cfg_.lr);
}

bool EndgameTrainer::sample_street_root(poker_ppo::PokerEnvironment& env,
                                        std::mt19937& rng, int target_round) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    env.reset();
    while (!env.is_terminal() && env.round() < target_round) {
        // mostly call/check, sometimes a raise of varied size: broad pot
        // coverage (solver leaf pots include multi-raise lines)
        int a = 1;
        if (u(rng) < 0.3) {
            static const int kRaises[] = {4, 7, 10};
            a = kRaises[rng() % 3];
        }
        auto mask = env.legal_action_mask();
        auto ma = mask.accessor<float, 1>();
        env.step(ma[a] > 0.5f ? a : 1);
    }
    return !env.is_terminal() && env.round() == target_round;
}

bool EndgameTrainer::river_sample_at(poker_ppo::PokerEnvironment& env,
                                     const uint8_t* b5, const HunlPBS& beta,
                                     std::vector<Sample>& out) {
    ExactStreetOracle exact(cfg_.t_river, cfg_.actions);
    std::array<std::vector<double>, 2> v;
    exact.value(env, b5, 5, beta, v);
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    if (contrib <= 0.0) return false;

    std::vector<uint8_t> v5;
    board_valid(b5, 5, v5);
    // normalized ranges for the FEATURES (the solver normalized its own copy)
    HunlPBS nb = beta;
    {
        double s0 = 0.0, s1 = 0.0;
        for (int i = 0; i < kCombos; ++i) {
            if (!v5[i]) {
                nb.r0[i] = nb.r1[i] = 0.0;
                continue;
            }
            s0 += nb.r0[i];
            s1 += nb.r1[i];
        }
        if (s0 <= 0.0 || s1 <= 0.0) return false;
        for (int i = 0; i < kCombos; ++i) {
            nb.r0[i] /= s0;
            nb.r1[i] /= s1;
        }
    }
    std::array<std::vector<double>, 2> omass;
    compat_mass(nb.r1, v5, omass[0]);
    compat_mass(nb.r0, v5, omass[1]);

    Sample smp;
    const double pot_leaf = 2.0 * contrib;
    smp.feat = HunlFeaturizer::features(3, b5, 5, pot_leaf, stack_, nb);
    smp.target.assign(2 * kCombos, 0.0f);
    smp.mask.assign(2 * kCombos, 0.0f);
    for (int p = 0; p < 2; ++p) {
        const auto& own = p == 0 ? nb.r0 : nb.r1;
        for (int i = 0; i < kCombos; ++i) {
            if (!v5[i] || own[i] <= 0.0 || omass[p][i] <= 0.0) continue;
            smp.target[p * kCombos + i] =
                static_cast<float>(v[p][i] / pot_leaf);   // pot units
            smp.mask[p * kCombos + i] = 1.0f;
        }
    }
    out.push_back(std::move(smp));
    return true;
}

void EndgameTrainer::direct_river_episode(poker_ppo::PokerEnvironment& env,
                                          std::mt19937& rng,
                                          std::vector<Sample>& fresh) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_street_root(env, rng, 3)) break;
    if (env.is_terminal() || env.round() != 3) return;
    uint8_t b5[5];
    for (int i = 0; i < 5; ++i)
        b5[i] = static_cast<uint8_t>(env.community_card(i));
    HunlPBS beta;
    beta.r0 = random_range(rng, b5, 5);
    beta.r1 = random_range(rng, b5, 5);
    river_sample_at(env, b5, beta, fresh);
}

void EndgameTrainer::gpu_river_epoch(
    std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>>& envs, int W,
    int ep, std::vector<Sample>& fresh, int episodes) {
    struct Pending {
        RiverSpec sp;
        std::string sig;
        bool ok = false;
    };
    std::vector<Pending> pend(static_cast<size_t>(episodes));
    std::unordered_map<std::string, TreeShape> shapes;
    std::mutex mtx;
    std::atomic<int> next{0};
    auto work = [&](int w) {
        std::mt19937 wrng(static_cast<unsigned>(
            cfg_.seed * 1000003u + ep * 7919u + w * 104729u + 17u));
        while (true) {
            const int e = next.fetch_add(1);
            if (e >= episodes) break;
            auto& env = *envs[w];
            bool root = false;
            for (int tries = 0; tries < 50 && !root; ++tries)
                root = sample_street_root(env, wrng, 3);
            if (!root) continue;
            Pending& p = pend[static_cast<size_t>(e)];
            for (int i = 0; i < 5; ++i)
                p.sp.board[i] = static_cast<uint8_t>(env.community_card(i));
            p.sp.r0 = random_range(wrng, p.sp.board.data(), 5);
            p.sp.r1 = random_range(wrng, p.sp.board.data(), 5);
            HunlPBS beta;
            beta.r0 = p.sp.r0;
            beta.r1 = p.sp.r1;
            HunlSolver tmp(env, beta, nullptr, cfg_.actions);
            auto sh = TreeShape::from(tmp);
            p.sig = sh.signature;
            for (const auto& nd : tmp.nodes()) {
                p.sp.node_contrib0.push_back(nd.contrib[0]);
                p.sp.node_contrib1.push_back(nd.contrib[1]);
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                shapes.emplace(p.sig, std::move(sh));
            }
            p.ok = true;
        }
    };
    if (W == 1) {
        work(0);
    } else {
        std::vector<std::thread> pool;
        for (int w = 0; w < W; ++w) pool.emplace_back(work, w);
        for (auto& th : pool) th.join();
    }

    // group by topology signature, solve in device batches
    std::unordered_map<std::string, std::vector<int>> groups;
    for (size_t e = 0; e < pend.size(); ++e)
        if (pend[e].ok) groups[pend[e].sig].push_back(static_cast<int>(e));
    for (auto& [sig, idxs] : groups) {
        for (size_t off = 0; off < idxs.size();
             off += static_cast<size_t>(cfg_.gpu_batch)) {
            const size_t end =
                std::min(idxs.size(), off + static_cast<size_t>(cfg_.gpu_batch));
            std::vector<RiverSpec> batch;
            batch.reserve(end - off);
            for (size_t k = off; k < end; ++k)
                batch.push_back(pend[static_cast<size_t>(idxs[k])].sp);
            BatchRiverSolver bs(shapes[sig], batch, device_, torch::kFloat);
            bs.solve(cfg_.t_river);
            std::vector<std::array<std::vector<double>, 2>> v, m;
            bs.root_values(v, m);
            for (size_t b = 0; b < batch.size(); ++b) {
                const RiverSpec& sp = batch[b];
                const double pot = 2.0 * sp.node_contrib0[0];
                if (pot <= 0.0) continue;
                // normalized ranges for the features
                std::vector<uint8_t> v5;
                board_valid(sp.board.data(), 5, v5);
                HunlPBS nb;
                nb.r0 = sp.r0;
                nb.r1 = sp.r1;
                double s0 = 0.0, s1 = 0.0;
                for (int i = 0; i < kCombos; ++i) {
                    if (!v5[i]) {
                        nb.r0[i] = nb.r1[i] = 0.0;
                        continue;
                    }
                    s0 += nb.r0[i];
                    s1 += nb.r1[i];
                }
                if (s0 <= 0.0 || s1 <= 0.0) continue;
                for (int i = 0; i < kCombos; ++i) {
                    nb.r0[i] /= s0;
                    nb.r1[i] /= s1;
                }
                Sample smp;
                smp.feat = HunlFeaturizer::features(3, sp.board.data(), 5,
                                                    pot, stack_, nb);
                smp.target.assign(2 * kCombos, 0.0f);
                smp.mask.assign(2 * kCombos, 0.0f);
                for (int p = 0; p < 2; ++p)
                    for (int i = 0; i < kCombos; ++i) {
                        if (m[b][p][i] < 0.5) continue;
                        smp.target[p * kCombos + i] =
                            static_cast<float>(v[b][p][i] / pot);
                        smp.mask[p * kCombos + i] = 1.0f;
                    }
                fresh.push_back(std::move(smp));
            }
        }
    }
}

std::vector<double> EndgameTrainer::random_range(std::mt19937& rng,
                                                 const uint8_t* board,
                                                 int nb) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // Solver-generated leaf ranges are STRUCTURED (sparse, correlated) —
    // iid noise ranges train a net that is useless at solver queries
    // (measured: heldout 0.005 on iid data, 0.15+ on turn-harvest data).
    // DeepStack's R(S,p), used here 70% of the time: recursively split the
    // valid combos into a WEAKER and a STRONGER half by hand strength on
    // this board, dividing the mass U(0,1) at each split (paper: "S1 =
    // weaker ⌊|S|/2⌋ hands, S2 = stronger"). A random-shuffle partition
    // reproduces the mass sparsity but not the strength polarization CFR
    // ranges actually have.
    if (u(rng) < 0.7) {
        std::vector<uint8_t> vmask;
        board_valid(board, nb, vmask);
        const auto& ct = ComboTable::get();
        std::vector<std::pair<double, int>> order;   // (strength, combo)
        order.reserve(kCombos);
        for (int i = 0; i < kCombos; ++i) {
            if (!vmask[i]) continue;
            double s;
            if (nb == 5) {
                s = static_cast<double>(combo_rank(i, board));
            } else if (nb == 4) {
                // turn: expected rank over river cards (DeepStack orders
                // earlier streets by expected hand strength)
                std::array<uint8_t, 5> b5{};
                for (int b = 0; b < nb; ++b) b5[b] = board[b];
                double acc = 0.0;
                int cnt = 0;
                for (int c = 0; c < kCards; ++c) {
                    bool dead = c == ct.cards[i][0] || c == ct.cards[i][1];
                    for (int b = 0; b < nb; ++b)
                        if (board[b] == c) dead = true;
                    if (dead) continue;
                    b5[nb] = static_cast<uint8_t>(c);
                    acc += static_cast<double>(combo_rank(i, b5.data()));
                    ++cnt;
                }
                s = cnt > 0 ? acc / cnt : 0.0;
            } else {
                // flop: sampled two-card runouts — ordering quality, not
                // exactness, is what the recursion needs
                std::array<uint8_t, 5> b5{};
                for (int b = 0; b < nb; ++b) b5[b] = board[b];
                double acc = 0.0;
                int cnt = 0;
                for (int s2 = 0; s2 < 8; ++s2) {
                    uint8_t t2, r2;
                    do {
                        t2 = static_cast<uint8_t>(rng() % kCards);
                    } while (t2 == ct.cards[i][0] || t2 == ct.cards[i][1] ||
                             t2 == board[0] || t2 == board[1] ||
                             t2 == board[2]);
                    do {
                        r2 = static_cast<uint8_t>(rng() % kCards);
                    } while (r2 == t2 || r2 == ct.cards[i][0] ||
                             r2 == ct.cards[i][1] || r2 == board[0] ||
                             r2 == board[1] || r2 == board[2]);
                    b5[3] = t2;
                    b5[4] = r2;
                    acc += static_cast<double>(combo_rank(i, b5.data()));
                    ++cnt;
                }
                s = cnt > 0 ? acc / cnt : 0.0;
            }
            order.emplace_back(s, i);
        }
        std::sort(order.begin(), order.end());
        std::vector<double> r(kCombos, 0.0);
        std::function<void(int, int, double)> split =
            [&](int lo, int hi, double mass) {
            if (hi - lo == 1) {
                r[static_cast<size_t>(order[static_cast<size_t>(lo)].second)] =
                    mass;
                return;
            }
            const int mid = lo + (hi - lo) / 2;
            const double q = u(rng);
            split(lo, mid, mass * q);
            split(mid, hi, mass * (1.0 - q));
        };
        if (!order.empty()) split(0, static_cast<int>(order.size()), 1.0);
        return r;
    }
    // iid coverage floor (broad but unstructured)
    const double k = 1.0 + 3.0 * u(rng);
    std::vector<double> r(kCombos);
    for (double& x : r) x = std::pow(u(rng), k);
    return r;
}

void EndgameTrainer::emit_root_sample(poker_ppo::PokerEnvironment& env,
                                      HunlSolver& s, const HunlPBS& beta,
                                      const uint8_t* board, int nb,
                                      int street, std::vector<Sample>& fresh) {
    std::array<std::vector<double>, 2> rv, rm;
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    if (contrib <= 0.0 || !s.avg_root_values(rv, rm)) return;
    HunlPBS nbeta = beta;
    std::vector<uint8_t> vmask;
    board_valid(board, nb, vmask);
    double s0 = 0.0, s1 = 0.0;
    for (int i = 0; i < kCombos; ++i) {
        if (!vmask[i]) {
            nbeta.r0[i] = nbeta.r1[i] = 0.0;
            continue;
        }
        s0 += nbeta.r0[i];
        s1 += nbeta.r1[i];
    }
    if (s0 <= 0.0 || s1 <= 0.0) return;
    for (int i = 0; i < kCombos; ++i) {
        nbeta.r0[i] /= s0;
        nbeta.r1[i] /= s1;
    }
    const double pot_root = 2.0 * contrib;
    Sample smp;
    smp.feat = HunlFeaturizer::features(street, board, nb, pot_root, stack_,
                                        nbeta);
    smp.target.assign(2 * kCombos, 0.0f);
    smp.mask.assign(2 * kCombos, 0.0f);
    for (int p = 0; p < 2; ++p) {
        const auto& own = p == 0 ? nbeta.r0 : nbeta.r1;
        for (int i = 0; i < kCombos; ++i) {
            if (rm[p][i] < 0.5 || own[i] <= 0.0) continue;
            smp.target[p * kCombos + i] =
                static_cast<float>(rv[p][i] / pot_root);
            smp.mask[p * kCombos + i] = 1.0f;
        }
    }
    fresh.push_back(std::move(smp));
}

void EndgameTrainer::self_play_episode(poker_ppo::PokerEnvironment& env,
                                       std::mt19937& rng,
                                       std::vector<Sample>& fresh) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_street_root(env, rng, 2)) break;
    if (env.is_terminal() || env.round() != 2) return;

    uint8_t b4[4];
    for (int i = 0; i < 4; ++i)
        b4[i] = static_cast<uint8_t>(env.community_card(i));
    HunlPBS beta;
    beta.r0 = random_range(rng, b4, 4);
    beta.r1 = random_range(rng, b4, 4);
    HunlNetOracle oracle(net_, stack_, device_);
    HunlSolver s(env, beta, &oracle, cfg_.actions);
    s.refresh_every = 5;
    s.track_root_values();

    std::uniform_int_distribution<int> dt(1, cfg_.t_turn);
    const int tstar = dt(rng);
    const int explorer = static_cast<int>(rng() & 1);
    int leaf = -1;
    uint8_t card = 0;
    HunlPBS leaf_beta;
    bool have = false;
    for (int t = 1; t <= cfg_.t_turn; ++t) {
        s.iterate(t);
        if (t == tstar)
            have = s.sample_leaf(rng, cfg_.eps_explore, explorer, &leaf,
                                 &card, &leaf_beta);
    }

    // Phase C: ReBeL's own value target for the solved subgame root —
    // (1/T)·Σ_t v^{π^t}(β_root), bootstrapped through the net's river
    // leaves. Street-2 rows teach the net TURN-ROOT values, the
    // prerequisite for flop solves with net leaves.
    emit_root_sample(env, s, beta, b4, 4, 2, fresh);

    // Algorithm 1's t*-sampled continuation leaf (ε-explored coverage)
    if (have) {
        std::array<uint8_t, 5> b5 = s.board();
        b5[s.board_count()] = card;
        env.push_state();
        for (int a : s.nodes()[leaf].path) env.step(a);
        river_sample_at(env, b5.data(), leaf_beta, fresh);
        env.pop_state();
    }

    // Harvest: additional (leaf, card) PBSs from the solver's own query
    // distribution (final-average beliefs) — each is one cheap river solve.
    const auto& nodes = s.nodes();
    std::vector<int> leaves;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].kind == HunlSolver::Node::StreetEnd)
            leaves.push_back(static_cast<int>(i));
    if (leaves.empty()) return;
    for (int h = 0; h < cfg_.harvest; ++h) {
        const int L = leaves[rng() % leaves.size()];
        // random candidate card off the board
        uint8_t c;
        while (true) {
            c = static_cast<uint8_t>(rng() % kCards);
            bool dead = false;
            for (int b = 0; b < s.board_count(); ++b)
                if (s.board()[b] == c) dead = true;
            if (!dead) break;
        }
        // final-average leaf beliefs, card-masked (river_sample_at
        // normalizes for the features; the solver normalizes its own copy)
        HunlPBS lb;
        {
            std::vector<double> r0, r1;
            // reaches under the final average profile at L
            // (private helper equivalent: rebuild via avg policies)
            r0 = beta.r0;
            r1 = beta.r1;
            int node = 0;
            for (int a : nodes[L].path) {
                const auto& nd = nodes[node];
                int k = 0;
                while (nd.acts[k] != a) ++k;
                std::vector<double>& mine = nd.player == 0 ? r0 : r1;
                for (int x = 0; x < kCombos; ++x)
                    if (mine[x] > 0.0)
                        mine[x] *= s.avg_policy(node, x)[k];
                node = nd.child[k];
            }
            lb.r0 = std::move(r0);
            lb.r1 = std::move(r1);
            const auto& ct = ComboTable::get();
            for (int i = 0; i < kCombos; ++i)
                if (ct.cards[i][0] == c || ct.cards[i][1] == c)
                    lb.r0[i] = lb.r1[i] = 0.0;
        }
        std::array<uint8_t, 5> b5 = s.board();
        b5[s.board_count()] = c;
        env.push_state();
        for (int a : nodes[L].path) env.step(a);
        river_sample_at(env, b5.data(), lb, fresh);
        env.pop_state();
    }
}

void EndgameTrainer::flop_episode(poker_ppo::PokerEnvironment& env,
                                  std::mt19937& rng,
                                  std::vector<Sample>& fresh) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_street_root(env, rng, 1)) break;
    if (env.is_terminal() || env.round() != 1) return;

    uint8_t b3[3];
    for (int i = 0; i < 3; ++i)
        b3[i] = static_cast<uint8_t>(env.community_card(i));
    HunlPBS beta;
    beta.r0 = random_range(rng, b3, 3);
    beta.r1 = random_range(rng, b3, 3);
    HunlNetOracle oracle(net_, stack_, device_);
    HunlSolver s(env, beta, &oracle, cfg_.actions);
    s.refresh_every = 10;   // net leaves at turn boards; play uses 10 too
    s.track_root_values();

    std::uniform_int_distribution<int> dt(1, cfg_.t_flop);
    const int tstar = dt(rng);
    const int explorer = static_cast<int>(rng() & 1);
    int leaf = -1;
    uint8_t card = 0;
    HunlPBS leaf_beta;
    bool have = false;
    for (int t = 1; t <= cfg_.t_flop; ++t) {
        s.iterate(t);
        if (t == tstar)
            have = s.sample_leaf(rng, cfg_.eps_explore, explorer, &leaf,
                                 &card, &leaf_beta);
    }

    // street-1 row: iterate-averaged FLOP-ROOT values, bootstrapped
    // through the net's turn leaves — the training signal a preflop
    // solver needs at ITS leaves.
    emit_root_sample(env, s, beta, b3, 3, 1, fresh);

    // a harvested (leaf, card) turn PBS becomes one turn solve whose
    // iterate-averaged root values are a street-2 row on the FLOP query
    // distribution (self-consistency: turn rows so far came from
    // random-range roots, not from flop-solve leaf beliefs).
    const auto& nodes = s.nodes();
    auto turn_root_sample = [&](int L, uint8_t c, const HunlPBS& lb) {
        std::array<uint8_t, 5> b4 = s.board();
        b4[s.board_count()] = c;
        env.push_state();
        for (int a : nodes[L].path) env.step(a);
        // board override: the env dealt its own turn card
        HunlSolver ts(env, lb, &oracle, cfg_.actions, b4.data(), 4);
        ts.refresh_every = 5;
        ts.track_root_values();
        const int T = std::max(1, cfg_.t_turn / 2);
        for (int t = 1; t <= T; ++t) ts.iterate(t);
        emit_root_sample(env, ts, lb, b4.data(), 4, 2, fresh);
        env.pop_state();
    };

    // Algorithm 1's t*-sampled continuation leaf (ε-explored coverage)
    if (have) turn_root_sample(leaf, card, leaf_beta);

    // final-average-belief harvest — turn solves are ~50× a river solve,
    // so take far fewer than the turn path's cfg_.harvest
    std::vector<int> leaves;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].kind == HunlSolver::Node::StreetEnd)
            leaves.push_back(static_cast<int>(i));
    if (leaves.empty()) return;
    const int H = std::max(1, std::min(cfg_.harvest, 2));
    for (int h = 0; h < H; ++h) {
        const int L = leaves[rng() % leaves.size()];
        uint8_t c;
        while (true) {
            c = static_cast<uint8_t>(rng() % kCards);
            bool dead = false;
            for (int b = 0; b < s.board_count(); ++b)
                if (s.board()[b] == c) dead = true;
            if (!dead) break;
        }
        HunlPBS lb;
        s.beliefs_at(L, lb.r0, lb.r1);
        const auto& ct = ComboTable::get();
        for (int i = 0; i < kCombos; ++i)
            if (ct.cards[i][0] == c || ct.cards[i][1] == c)
                lb.r0[i] = lb.r1[i] = 0.0;
        turn_root_sample(L, c, lb);
    }
}

void EndgameTrainer::replay_insert(Sample&& smp) {
    ++seen_;
    if (static_cast<int>(replay_.size()) < cfg_.replay_cap) {
        replay_.push_back(std::move(smp));
    } else if (circular_) {
        // ReBeL: circular buffer — overwrite the oldest (recency window,
        // tracks the net's evolving query distribution)
        replay_[static_cast<size_t>((seen_ - 1) % cfg_.replay_cap)] =
            std::move(smp);
    } else {
        // reservoir — uniform over all samples ever seen (stationary data)
        std::uniform_int_distribution<long> d(0, seen_ - 1);
        const long j = d(rng_);
        if (j < static_cast<long>(replay_.size()))
            replay_[static_cast<size_t>(j)] = std::move(smp);
    }
}

void EndgameTrainer::print_pot_hist(const std::vector<Sample>& v,
                                    const char* tag) {
    if (v.empty()) return;
    // feat[4 + kCards] = pot/stack (see HunlFeaturizer)
    static const double kEdge[] = {0.05, 0.10, 0.20, 0.40, 0.70};
    long h[6] = {};
    for (const Sample& s : v) {
        const double p = s.feat[4 + kCards];
        int b = 0;
        while (b < 5 && p >= kEdge[b]) ++b;
        ++h[b];
    }
    const double n = static_cast<double>(v.size());
    std::printf("  [%s] pot/stack: <5%%:%.0f%%  5-10:%.0f%%  10-20:%.0f%%  "
                "20-40:%.0f%%  40-70:%.0f%%  70+:%.0f%%  (n=%zu)\n",
                tag, 100.0 * h[0] / n, 100.0 * h[1] / n, 100.0 * h[2] / n,
                100.0 * h[3] / n, 100.0 * h[4] / n, 100.0 * h[5] / n,
                v.size());
}

void EndgameTrainer::append_dataset(const std::string& path,
                                    const std::vector<Sample>& fresh) {
    if (fresh.empty()) return;
    // single-writer guard: concurrent appenders interleave buffered
    // 40KB rows and corrupt the file structurally (observed on a rental:
    // a double-launched generator, one dataset). flock held for process
    // lifetime; a second writer exits loudly instead.
    {
        static std::mutex lk_mtx;
        static std::unordered_map<std::string, int> lk_fds;
        std::lock_guard<std::mutex> g(lk_mtx);
        if (lk_fds.find(path) == lk_fds.end()) {
            const int fd =
                ::open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
            if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
                std::fprintf(stderr,
                             "  [data] %s is being appended by ANOTHER "
                             "process — refusing to interleave (this "
                             "corrupts rows). exiting\n", path.c_str());
                std::exit(1);
            }
            lk_fds[path] = fd;
        }
    }
    const int n_out = 2 * kCombos;
    const bool empty_file = !std::filesystem::exists(path) ||
                            std::filesystem::file_size(path) == 0;
    if (!empty_file && !check_data_header(path, n_out)) {
        std::fprintf(stderr,
                     "  [data] %s has a foreign header — refusing to append "
                     "(featurizer change? move the old file aside)\n",
                     path.c_str());
        std::exit(1);
    }
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f) {
        std::fprintf(stderr, "  [data] cannot open %s for append\n",
                     path.c_str());
        return;
    }
    if (empty_file) {
        const uint32_t hdr[4] = {kDataMagic, kDataVersion,
                                 static_cast<uint32_t>(HunlFeaturizer::kDim),
                                 static_cast<uint32_t>(n_out)};
        f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    }
    std::vector<uint8_t> m8(static_cast<size_t>(n_out));
    for (const Sample& s : fresh) {
        f.write(reinterpret_cast<const char*>(s.feat.data()),
                sizeof(float) * HunlFeaturizer::kDim);
        f.write(reinterpret_cast<const char*>(s.target.data()),
                sizeof(float) * static_cast<size_t>(n_out));
        for (int j = 0; j < n_out; ++j) m8[j] = s.mask[j] > 0.5f ? 1 : 0;
        f.write(reinterpret_cast<const char*>(m8.data()), n_out);
    }
}

void EndgameTrainer::load_dataset(const std::string& path) {
    const int n_out = 2 * kCombos;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("  [data] %s not found — starting empty\n", path.c_str());
        return;
    }
    if (!check_data_header(path, n_out)) {
        std::fprintf(stderr,
                     "  [data] %s header mismatch (featurizer or format "
                     "changed) — regenerate or convert\n",
                     path.c_str());
        std::exit(1);
    }
    f.seekg(4 * sizeof(uint32_t));
    std::vector<uint8_t> m8(static_cast<size_t>(n_out));
    long rows = 0;
    while (true) {
        Sample s;
        s.feat.resize(HunlFeaturizer::kDim);
        s.target.resize(static_cast<size_t>(n_out));
        s.mask.resize(static_cast<size_t>(n_out));
        if (!f.read(reinterpret_cast<char*>(s.feat.data()),
                    sizeof(float) * HunlFeaturizer::kDim)) {
            if (f.gcount() != 0)
                std::fprintf(stderr, "  [data] %s: truncated row %ld — "
                             "keeping what loaded\n", path.c_str(), rows);
            break;
        }
        if (!f.read(reinterpret_cast<char*>(s.target.data()),
                    sizeof(float) * static_cast<size_t>(n_out)) ||
            !f.read(reinterpret_cast<char*>(m8.data()), n_out)) {
            std::fprintf(stderr, "  [data] %s: truncated row %ld — keeping "
                         "what loaded\n", path.c_str(), rows);
            break;
        }
        for (int j = 0; j < n_out; ++j) s.mask[j] = m8[j] ? 1.0f : 0.0f;
        // fixed, config-independent heldout slice: same rows for every
        // run off this file, so heldout numbers compare across nets
        if (rows % 50 == 0 && heldout_.size() < 8192)
            heldout_.push_back(std::move(s));
        else
            replay_insert(std::move(s));
        ++rows;
    }
    // masked target variance of the heldout slice = the "predict the mean"
    // MSE baseline that makes heldout readable as explained variance
    double sum = 0.0, sq = 0.0, cnt = 0.0;
    for (const Sample& s : heldout_)
        for (int j = 0; j < n_out; ++j)
            if (s.mask[j] > 0.5f) {
                sum += s.target[j];
                sq += static_cast<double>(s.target[j]) * s.target[j];
                cnt += 1.0;
            }
    const double var =
        cnt > 0 ? sq / cnt - (sum / cnt) * (sum / cnt) : 0.0;
    std::printf("  [data] %s: %ld rows -> replay %zu, heldout %zu "
                "(target var %.3e)\n",
                path.c_str(), rows, replay_.size(), heldout_.size(), var);
    // Zero-sum residual of the TARGETS under the same weights the net's
    // outer correction uses (Σ r0·m0·v0 + Σ r1·m1·v1, per-entry units
    // s/2Z). Exact solver values satisfy the identity up to CFR
    // convergence error — a large residual here means the correction
    // formula is wrong, and it would push the net AWAY from the targets.
    {
        const auto& ct = ComboTable::get();
        const int off = 4 + kCards + 1;
        double mean_c = 0.0, max_c = 0.0;
        long n = 0;
        for (const Sample& smp : heldout_) {
            const float* r0 = smp.feat.data() + off;
            const float* r1 = smp.feat.data() + off + kCombos;
            double a0[kCards] = {}, a1[kCards] = {};
            for (int i = 0; i < kCombos; ++i) {
                a0[ct.cards[i][0]] += r0[i];
                a0[ct.cards[i][1]] += r0[i];
                a1[ct.cards[i][0]] += r1[i];
                a1[ct.cards[i][1]] += r1[i];
            }
            double S0 = 0.0, S1 = 0.0;
            for (int i = 0; i < kCombos; ++i) {
                S0 += r0[i];
                S1 += r1[i];
            }
            double s = 0.0, Z = 0.0;
            for (int i = 0; i < kCombos; ++i) {
                const double m0 =
                    S1 - a1[ct.cards[i][0]] - a1[ct.cards[i][1]] + r1[i];
                const double m1 =
                    S0 - a0[ct.cards[i][0]] - a0[ct.cards[i][1]] + r0[i];
                s += r0[i] * m0 * smp.target[i] +
                     r1[i] * m1 * smp.target[kCombos + i];
                Z += r0[i] * m0;
            }
            if (Z <= 0.0) continue;
            const double c = std::fabs(s) / (2.0 * Z);
            mean_c += c;
            max_c = std::max(max_c, c);
            ++n;
        }
        if (n > 0)
            std::printf("  [data] target zero-sum residual |s|/2Z: "
                        "mean %.2e max %.2e (pot units)\n",
                        mean_c / n, max_c);
    }
    print_pot_hist(replay_, "data");
}

double EndgameTrainer::train_net() {
    if (replay_.empty()) return 0.0;
    const int n_out = 2 * kCombos;
    std::uniform_int_distribution<size_t> pick(0, replay_.size() - 1);
    double last = 0.0;
    for (int step = 0; step < cfg_.sgd_steps; ++step) {
        const int B = std::min<int>(cfg_.batch,
                                    static_cast<int>(replay_.size()));
        auto X = torch::empty({B, HunlFeaturizer::kDim}, torch::kFloat);
        auto Y = torch::empty({B, n_out}, torch::kFloat);
        auto M = torch::empty({B, n_out}, torch::kFloat);
        for (int b = 0; b < B; ++b) {
            const Sample& s = replay_[pick(rng_)];
            std::copy(s.feat.begin(), s.feat.end(),
                      X.data_ptr<float>() +
                          static_cast<size_t>(b) * HunlFeaturizer::kDim);
            std::copy(s.target.begin(), s.target.end(),
                      Y.data_ptr<float>() + static_cast<size_t>(b) * n_out);
            std::copy(s.mask.begin(), s.mask.end(),
                      M.data_ptr<float>() + static_cast<size_t>(b) * n_out);
        }
        opt_->zero_grad();
        auto Xd = X.to(device_), Yd = Y.to(device_), Md = M.to(device_);
        auto err = net_->forward(Xd) - Yd;
        torch::Tensor pt;
        if (cfg_.huber_delta > 0.0) {
            // pointwise Huber (both papers): quadratic below delta,
            // linear above — the rare small-pot all-in rows with
            // |target| ~ stack/pot stop dominating the gradient
            const double d = cfg_.huber_delta;
            auto a = err.abs();
            pt = torch::where(a <= d, 0.5 * err * err, d * (a - 0.5 * d));
        } else {
            pt = err * err;
        }
        auto loss = (pt * Md).sum() / Md.sum().clamp_min(1.0);
        loss.backward();
        opt_->step();
        if (step == cfg_.sgd_steps - 1) last = loss.item<double>();
    }
    return last;
}

double EndgameTrainer::heldout_mse(const std::vector<Sample>& fresh) {
    if (fresh.empty()) return 0.0;
    torch::NoGradGuard ng;
    const long N = static_cast<long>(fresh.size());
    auto X = torch::empty({N, HunlFeaturizer::kDim}, torch::kFloat);
    for (long r = 0; r < N; ++r)
        std::copy(fresh[r].feat.begin(), fresh[r].feat.end(),
                  X.data_ptr<float>() +
                      static_cast<size_t>(r) * HunlFeaturizer::kDim);
    auto y = net_->forward(X.to(device_)).to(torch::kCPU).contiguous();
    auto acc = y.accessor<float, 2>();
    double se = 0.0, cnt = 0.0;
    for (long r = 0; r < N; ++r)
        for (int j = 0; j < 2 * kCombos; ++j)
            if (fresh[r].mask[j] > 0.5f) {
                const double e = acc[r][j] - fresh[r].target[j];
                se += e * e;
                cnt += 1.0;
            }
    return cnt > 0 ? se / cnt : 0.0;
}

double EndgameTrainer::probe_turn_expl(HunlValueOracle* oracle, int T,
                                       int refresh_every,
                                       uint32_t situation_seed) {
    // own env: board, prefix and ranges are all functions of the seed alone
    poker_ppo::PokerEnvironment env(poker_ppo::kPokerConfig,
                                    poker_ppo::config::kBetConfig,
                                    situation_seed);
    std::mt19937 fixed(situation_seed);   // fixed probe situation per seed
    for (int tries = 0; tries < 50; ++tries)
        if (sample_street_root(env, fixed, 2)) break;
    uint8_t b4[4];
    for (int i = 0; i < 4; ++i)
        b4[i] = static_cast<uint8_t>(env.community_card(i));
    HunlPBS beta;
    beta.r0 = random_range(fixed, b4, 4);
    beta.r1 = random_range(fixed, b4, 4);
    HunlSolver s(env, beta, oracle, cfg_.actions);
    s.refresh_every = refresh_every;
    for (int t = 1; t <= T; ++t) s.iterate(t);
    // the TRUE judge: composed two-street BR (river deviations allowed)
    return s.exploitability_composed(cfg_.t_river / 2);
}

void EndgameTrainer::run() {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    poker_ppo::PokerEnvironment env(poker_ppo::kPokerConfig,
                                    poker_ppo::config::kBetConfig,
                                    cfg_.seed + 99);
    const double bb =
        static_cast<double>(env.game_config().big_blind);
    int W = cfg_.threads > 0
        ? cfg_.threads
        : static_cast<int>(std::thread::hardware_concurrency());
    W = std::max(1, std::min(W, cfg_.episodes));
    if (W > 1) torch::set_num_threads(1);   // workers ARE the parallelism
    std::printf("HUNL endgame ReBeL (%s): T_turn=%d T_river=%d episodes=%d "
                "epochs=%d hidden=%dx%d%s sgd=%d batch=%d lr=%g",
                cfg_.river_only ? "direct-river"
                                : cfg_.flop_mode ? "flop-selfplay"
                                                 : "turn-selfplay",
                cfg_.t_turn, cfg_.t_river, cfg_.episodes, cfg_.epochs,
                cfg_.hidden, cfg_.layers, cfg_.gelu_ln ? "(gelu+ln)" : "",
                cfg_.sgd_steps, cfg_.batch, cfg_.lr);
    if (cfg_.flop_mode) std::printf(" T_flop=%d", cfg_.t_flop);
    if (cfg_.lr_final > 0.0) std::printf("->%g", cfg_.lr_final);
    if (cfg_.huber_delta > 0.0)
        std::printf(" loss=huber(%g)", cfg_.huber_delta);
    else
        std::printf(" loss=mse");
    std::printf(" buf=%s%s", circular_ ? "circular" : "reservoir",
                cfg_.zero_sum ? "" : " zero_sum=off");
    std::printf(" harvest=%d device=%s workers=%d\n", cfg_.harvest,
                device_.is_cuda() ? "cuda" : "cpu", W);
    if (cfg_.episodes == 0)
        std::printf("  [offline] no sampling — epochs are pure SGD on the "
                    "loaded dataset\n");
    // per-worker envs + HandRanks pre-warm (lazy 120MB load isn't racy
    // only because we touch it before spawning)
    std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>> envs;
    for (int w = 0; w < W; ++w)
        envs.push_back(std::make_unique<poker_ppo::PokerEnvironment>(
            poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
            cfg_.seed + 1000 + w));
    // GPU pipeline: its spec builders get their own envs so the GPU
    // thread and the CPU worker pool can run the SAME epoch concurrently
    std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>> envs_gpu;
    const int Wg = 4;
    double gpu_frac = 0.33;   // ≈ 53 / (53 + 110) measured split
    if (cfg_.river_only && cfg_.gpu_batch > 0) {
        if (const char* s = std::getenv("REBEL_GPU_FRAC"))
            gpu_frac = std::atof(s);
        for (int w = 0; w < Wg; ++w)
            envs_gpu.push_back(std::make_unique<poker_ppo::PokerEnvironment>(
                poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
                cfg_.seed + 9000 + w));
    }
    {
        const uint8_t warm[5] = {0, 5, 10, 15, 20};
        (void)combo_rank(ComboTable::get().id[25][30], warm);
    }
    if (!cfg_.data_in.empty()) load_dataset(cfg_.data_in);
    // reference: the same probe turn solves with EXACT leaves (slow, once).
    // one situation per seed; the net probe below averages the SAME seeds.
    auto probe_seed = [](int k) { return 12345u + 1000u * static_cast<unsigned>(k); };
    for (int k = 0; k < cfg_.probe_k; ++k) {
        auto t0 = clock::now();
        ExactStreetOracle exact(cfg_.t_river / 2, cfg_.actions);
        const double ref = probe_turn_expl(&exact, /*T=*/30,
                                           /*refresh_every=*/10,
                                           probe_seed(k));
        std::printf("  probe[%d] turn expl (exact leaves, T=30): %.4f bb  "
                    "(%.0fs)\n", k, ref / bb, secs(t0, clock::now()));
        std::fflush(stdout);
    }

    for (int ep = 1; ep <= cfg_.epochs; ++ep) {
        auto t_sp0 = clock::now();
        if (cfg_.lr_final > 0.0 && cfg_.epochs > 1) {
            const double a = static_cast<double>(ep - 1) / (cfg_.epochs - 1);
            const double cur = cfg_.lr + (cfg_.lr_final - cfg_.lr) * a;
            for (auto& g : opt_->param_groups())
                static_cast<torch::optim::AdamOptions&>(g.options()).lr(cur);
        }
        if (cfg_.river_only && cfg_.gpu_batch > 0 && ep == 1 &&
            !device_.is_cuda())
            std::fprintf(stderr,
                         "  [warn] REBEL_GPU_BATCH on a CPU-torch device is "
                         "~100x SLOWER than the worker path — use it on "
                         "CUDA only\n");
        // GPU pipeline share: a device thread solves e_gpu targets in
        // lockstep batches while the CPU pool solves the rest — separate
        // resources, so sequential use (the old exclusive gpu path)
        // wasted whichever one wasn't running
        const int e_gpu =
            (cfg_.river_only && cfg_.gpu_batch > 0 && device_.is_cuda())
                ? std::min(cfg_.episodes,
                           static_cast<int>(cfg_.episodes * gpu_frac))
                : 0;
        const int e_cpu = cfg_.episodes - e_gpu;
        std::vector<Sample> fresh_gpu;
        std::unique_ptr<std::thread> gpu_thr;
        if (e_gpu > 0)
            gpu_thr = std::make_unique<std::thread>([&, ep, e_gpu] {
                gpu_river_epoch(envs_gpu, Wg, ep, fresh_gpu, e_gpu);
            });
        // episodes fan out over the worker pool (work-stealing counter)
        std::vector<std::vector<Sample>> fresh_w(W);
        std::atomic<int> next{0};
        auto work = [&](int w) {
            std::mt19937 wrng(static_cast<unsigned>(
                cfg_.seed * 1000003u + ep * 7919u + w * 104729u));
            while (true) {
                const int e = next.fetch_add(1);
                if (e >= e_cpu) break;
                if (cfg_.river_only)
                    direct_river_episode(*envs[w], wrng, fresh_w[w]);
                else if (cfg_.flop_mode)
                    flop_episode(*envs[w], wrng, fresh_w[w]);
                else
                    self_play_episode(*envs[w], wrng, fresh_w[w]);
            }
        };
        if (W == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(W);
            for (int w = 0; w < W; ++w) pool.emplace_back(work, w);
            for (auto& th : pool) th.join();
        }
        std::vector<Sample> fresh;
        for (auto& fw : fresh_w)
            for (auto& s : fw) fresh.push_back(std::move(s));
        if (gpu_thr) {
            gpu_thr->join();
            for (auto& s : fresh_gpu) fresh.push_back(std::move(s));
        }
        auto t_sp1 = clock::now();
        if (ep == 1) print_pot_hist(fresh, "gen");
        if (!cfg_.data_out.empty()) append_dataset(cfg_.data_out, fresh);
        // out-of-sample probe: the persistent split when a dataset was
        // loaded (comparable across configs), else this epoch's fresh rows
        // BEFORE they are trained on
        const double heldout = !heldout_.empty() ? heldout_mse(heldout_)
                                                 : heldout_mse(fresh);
        for (Sample& smp : fresh) replay_insert(std::move(smp));
        auto t_tr0 = clock::now();
        const double vloss = train_net();
        auto t_tr1 = clock::now();
        double net_expl = -1.0, t_probe = 0.0;
        if (cfg_.probe_k > 0 && (ep % 5 == 0 || ep == cfg_.epochs)) {
            auto t_pr0 = clock::now();
            HunlNetOracle no(net_, stack_, device_);
            double sum = 0.0;
            for (int k = 0; k < cfg_.probe_k; ++k)
                sum += probe_turn_expl(&no, /*T=*/30,
                                       /*refresh_every=*/1, probe_seed(k));
            net_expl = sum / cfg_.probe_k;
            t_probe = secs(t_pr0, clock::now());
        }
        std::printf("  epoch %3d  replay=%5zu  vloss=%.3e  heldout=%.3e"
                    "  [sp %.0fs train %.0fs",
                    ep, replay_.size(), vloss, heldout,
                    secs(t_sp0, t_sp1), secs(t_tr0, t_tr1));
        if (net_expl >= 0.0)
            std::printf(" probe %.0fs]  probe-turn-expl(net)=%.4f bb",
                        t_probe, net_expl / bb);
        else
            std::printf("]");
        std::printf("\n");
        std::fflush(stdout);
        if (!cfg_.ckpt.empty()) torch::save(net_, cfg_.ckpt);
    }
}

}  // namespace rebel_hunl
