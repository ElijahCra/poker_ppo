#include "hunl_value.h"

#include <algorithm>
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
    const double contrib =
        static_cast<double>(env.game_config().initial_stack) - env.stack(0);
    const double pot = 2.0 * contrib;
    torch::NoGradGuard ng;
    auto f = HunlFeaturizer::features(street_of(nb), board, nb, pot, stack_,
                                      beta);
    auto x = torch::from_blob(f.data(), {1, HunlFeaturizer::kDim},
                              torch::kFloat)
                 .clone();
    auto y = net_->forward(x).squeeze(0);
    auto acc = y.accessor<float, 1>();
    // net predicts values in POT units (paper normalization: well-
    // conditioned across pot sizes; future betting can push |v| past 1)
    for (int p = 0; p < 2; ++p) {
        out[p].assign(kCombos, 0.0);
        for (int i = 0; i < kCombos; ++i)
            out[p][i] = static_cast<double>(acc[p * kCombos + i]) * pot;
    }
}

EndgameTrainer::EndgameTrainer(EndgameConfig cfg)
    : cfg_(std::move(cfg)),
      stack_(static_cast<double>(
          poker_ppo::kPokerConfig.game.initial_stack)),
      rng_(static_cast<unsigned>(cfg_.seed)) {
    torch::manual_seed(cfg_.seed);
    net_ = HunlValueNet(cfg_.hidden);
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
                                       std::mt19937& rng) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_turn_root(env, rng)) break;
    if (env.is_terminal() || env.round() != 2) return;

    HunlPBS beta;
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    HunlNetOracle oracle(net_, stack_);
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
        auto loss =
            ((net_->forward(X) - Y).pow(2) * M).sum() / M.sum().clamp_min(1.0);
        loss.backward();
        opt_->step();
        if (step == cfg_.sgd_steps - 1) last = loss.item<double>();
    }
    return last;
}

double EndgameTrainer::probe_mse(int k) {
    if (replay_.empty()) return 0.0;
    torch::NoGradGuard ng;
    std::uniform_int_distribution<size_t> pick(0, replay_.size() - 1);
    double se = 0.0, cnt = 0.0;
    for (int i = 0; i < k; ++i) {
        const Sample& s = replay_[pick(rng_)];
        auto x = torch::from_blob(const_cast<float*>(s.feat.data()),
                                  {1, HunlFeaturizer::kDim}, torch::kFloat);
        auto y = net_->forward(x).squeeze(0);
        auto acc = y.accessor<float, 1>();
        for (int j = 0; j < 2 * kCombos; ++j)
            if (s.mask[j] > 0.5f) {
                const double e = acc[j] - s.target[j];
                se += e * e;
                cnt += 1.0;
            }
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
    poker_ppo::PokerEnvironment env(poker_ppo::kPokerConfig,
                                    poker_ppo::config::kBetConfig,
                                    cfg_.seed + 99);
    const double bb =
        static_cast<double>(env.game_config().big_blind);
    std::printf("HUNL endgame ReBeL: T_turn=%d T_river=%d episodes=%d "
                "epochs=%d hidden=%d\n",
                cfg_.t_turn, cfg_.t_river, cfg_.episodes, cfg_.epochs,
                cfg_.hidden);
    // reference: the same probe turn solve with EXACT leaves (slow, once)
    ExactStreetOracle exact(cfg_.t_river / 2, cfg_.actions);
    const double ref =
        probe_turn_expl(env, &exact, /*T=*/30, /*refresh_every=*/10);
    std::printf("  probe turn expl (exact leaves, T=30): %.4f bb\n",
                ref / bb);
    std::fflush(stdout);

    for (int ep = 1; ep <= cfg_.epochs; ++ep) {
        for (int e = 0; e < cfg_.episodes; ++e)
            self_play_episode(env, rng_);
        const double vloss = train_net();
        const double probe = probe_mse(cfg_.probe_k);
        double net_expl = -1.0;
        if (ep % 5 == 0 || ep == cfg_.epochs) {
            HunlNetOracle no(net_, stack_);
            net_expl = probe_turn_expl(env, &no, /*T=*/30,
                                       /*refresh_every=*/1);
        }
        std::printf("  epoch %3d  replay=%5zu  vloss=%.6f  probe=%.6f",
                    ep, replay_.size(), vloss, probe);
        if (net_expl >= 0.0)
            std::printf("  probe-turn-expl(net)=%.4f bb", net_expl / bb);
        std::printf("\n");
        std::fflush(stdout);
    }
}

}  // namespace rebel_hunl
