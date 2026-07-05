#include "hunl_value.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>

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
}  // namespace

std::vector<float> HunlFeaturizer::features(int street, const uint8_t* board,
                                            int nb, double pot, double stack,
                                            const HunlPBS& beta) {
    std::vector<float> f(kDim, 0.0f);
    f[street] = 1.0f;
    for (int i = 0; i < nb; ++i) f[4 + board[i]] = 1.0f;
    f[4 + kCards] = static_cast<float>(pot / stack);
    const int off = 4 + kCards + 1;
    for (int i = 0; i < kCombos; ++i) {
        f[off + i] = static_cast<float>(beta.r0[i]);
        f[off + kCombos + i] = static_cast<float>(beta.r1[i]);
    }
    return f;
}

HunlValueNetImpl::HunlValueNetImpl(int hidden) {
    l1 = register_module("l1", torch::nn::Linear(HunlFeaturizer::kDim, hidden));
    l2 = register_module("l2", torch::nn::Linear(hidden, hidden));
    l3 = register_module("l3", torch::nn::Linear(hidden, 2 * kCombos));
}

torch::Tensor HunlValueNetImpl::forward(torch::Tensor x) {
    x = torch::relu(l1(x));
    x = torch::relu(l2(x));
    return l3(x);
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
    for (long r = 0; r < N; ++r) {
        b[nb_base] = cards[static_cast<size_t>(r)];
        auto f = HunlFeaturizer::features(street_of(nb_base + 1), b.data(),
                                          nb_base + 1, pot, stack_,
                                          betas[static_cast<size_t>(r)]);
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

EndgameTrainer::EndgameTrainer(EndgameConfig cfg)
    : cfg_(std::move(cfg)),
      stack_(static_cast<double>(
          poker_ppo::kPokerConfig.game.initial_stack)),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      rng_(static_cast<unsigned>(cfg_.seed)) {
    torch::manual_seed(cfg_.seed);
    net_ = HunlValueNet(cfg_.hidden);
    net_->to(device_);
    opt_ = std::make_unique<torch::optim::Adam>(net_->parameters(), cfg_.lr);
}

bool EndgameTrainer::sample_turn_root(poker_ppo::PokerEnvironment& env,
                                      std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    env.reset();
    while (!env.is_terminal() && env.round() < 2) {
        // mostly call/check, sometimes pot-raise: varied pots, rare all-ins
        const int a = u(rng) < 0.25 ? 7 : 1;
        auto mask = env.legal_action_mask();
        auto ma = mask.accessor<float, 1>();
        env.step(ma[a] > 0.5f ? a : 1);
    }
    return !env.is_terminal() && env.round() == 2;
}

std::vector<double> EndgameTrainer::random_range(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // per-episode sharpness varies coverage of the range simplex
    const double k = 1.0 + 3.0 * u(rng);
    std::vector<double> r(kCombos);
    for (double& x : r) x = std::pow(u(rng), k);
    return r;
}

void EndgameTrainer::self_play_episode(poker_ppo::PokerEnvironment& env,
                                       std::mt19937& rng,
                                       std::vector<Sample>& fresh) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_turn_root(env, rng)) break;
    if (env.is_terminal() || env.round() != 2) return;

    HunlPBS beta;
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    HunlNetOracle oracle(net_, stack_, device_);
    HunlSolver s(env, beta, &oracle, cfg_.actions);
    s.refresh_every = 5;

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
    if (!have) return;   // walk ended at an in-street terminal

    // exact river solve at the sampled leaf = the grounded target
    std::array<uint8_t, 5> b5 = s.board();
    b5[s.board_count()] = card;
    env.push_state();
    for (int a : s.nodes()[leaf].path) env.step(a);
    ExactStreetOracle exact(cfg_.t_river, cfg_.actions);
    std::array<std::vector<double>, 2> v;
    exact.value(env, b5.data(), 5, leaf_beta, v);
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    env.pop_state();

    // masks: valid on the full board, own mass, opponent compatible mass
    std::vector<uint8_t> v5;
    board_valid(b5.data(), 5, v5);
    std::array<std::vector<double>, 2> omass;
    compat_mass(leaf_beta.r1, v5, omass[0]);
    compat_mass(leaf_beta.r0, v5, omass[1]);

    Sample smp;
    smp.feat = HunlFeaturizer::features(3, b5.data(), 5, 2.0 * contrib,
                                        stack_, leaf_beta);
    smp.target.assign(2 * kCombos, 0.0f);
    smp.mask.assign(2 * kCombos, 0.0f);
    const double pot_leaf = 2.0 * contrib;
    for (int p = 0; p < 2; ++p) {
        const auto& own = p == 0 ? leaf_beta.r0 : leaf_beta.r1;
        for (int i = 0; i < kCombos; ++i) {
            if (!v5[i] || own[i] <= 0.0 || omass[p][i] <= 0.0) continue;
            smp.target[p * kCombos + i] =
                static_cast<float>(v[p][i] / pot_leaf);   // pot units
            smp.mask[p * kCombos + i] = 1.0f;
        }
    }
    fresh.push_back(std::move(smp));   // probed out-of-sample, then merged
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
        auto loss = ((net_->forward(Xd) - Yd).pow(2) * Md).sum() /
                    Md.sum().clamp_min(1.0);
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

double EndgameTrainer::probe_turn_expl(poker_ppo::PokerEnvironment& env,
                                       HunlValueOracle* oracle, int T,
                                       int refresh_every) {
    std::mt19937 fixed(12345);          // fixed probe situation
    for (int tries = 0; tries < 50; ++tries)
        if (sample_turn_root(env, fixed)) break;
    HunlPBS beta;
    beta.r0 = random_range(fixed);
    beta.r1 = random_range(fixed);
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
    std::printf("HUNL endgame ReBeL: T_turn=%d T_river=%d episodes=%d "
                "epochs=%d hidden=%d device=%s workers=%d\n",
                cfg_.t_turn, cfg_.t_river, cfg_.episodes, cfg_.epochs,
                cfg_.hidden, device_.is_cuda() ? "cuda" : "cpu", W);
    // per-worker envs + HandRanks pre-warm (lazy 120MB load isn't racy
    // only because we touch it before spawning)
    std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>> envs;
    for (int w = 0; w < W; ++w)
        envs.push_back(std::make_unique<poker_ppo::PokerEnvironment>(
            poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
            cfg_.seed + 1000 + w));
    {
        const uint8_t warm[5] = {0, 5, 10, 15, 20};
        (void)combo_rank(ComboTable::get().id[25][30], warm);
    }
    // reference: the same probe turn solve with EXACT leaves (slow, once)
    auto t0 = clock::now();
    ExactStreetOracle exact(cfg_.t_river / 2, cfg_.actions);
    const double ref =
        probe_turn_expl(env, &exact, /*T=*/30, /*refresh_every=*/10);
    std::printf("  probe turn expl (exact leaves, T=30): %.4f bb  (%.0fs)\n",
                ref / bb, secs(t0, clock::now()));
    std::fflush(stdout);

    for (int ep = 1; ep <= cfg_.epochs; ++ep) {
        auto t_sp0 = clock::now();
        // episodes fan out over the worker pool (work-stealing counter)
        std::vector<std::vector<Sample>> fresh_w(W);
        std::atomic<int> next{0};
        auto work = [&](int w) {
            std::mt19937 wrng(static_cast<unsigned>(
                cfg_.seed * 1000003u + ep * 7919u + w * 104729u));
            while (true) {
                const int e = next.fetch_add(1);
                if (e >= cfg_.episodes) break;
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
        auto t_sp1 = clock::now();
        // out-of-sample probe BEFORE these rows are trained on
        const double heldout = heldout_mse(fresh);
        for (Sample& smp : fresh) {
            ++seen_;
            if (static_cast<int>(replay_.size()) < cfg_.replay_cap) {
                replay_.push_back(std::move(smp));
            } else {
                std::uniform_int_distribution<long> d(0, seen_ - 1);
                const long j = d(rng_);
                if (j < static_cast<long>(replay_.size()))
                    replay_[static_cast<size_t>(j)] = std::move(smp);
            }
        }
        auto t_tr0 = clock::now();
        const double vloss = train_net();
        auto t_tr1 = clock::now();
        double net_expl = -1.0, t_probe = 0.0;
        if (ep % 5 == 0 || ep == cfg_.epochs) {
            auto t_pr0 = clock::now();
            HunlNetOracle no(net_, stack_, device_);
            net_expl = probe_turn_expl(env, &no, /*T=*/30,
                                       /*refresh_every=*/1);
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
    }
}

}  // namespace rebel_hunl
