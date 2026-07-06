#include "hunl_value.h"

#include "hunl_gpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <filesystem>
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
    net_ = HunlValueNet(cfg_.hidden);
    if (!cfg_.ckpt.empty() && std::filesystem::exists(cfg_.ckpt)) {
        torch::load(net_, cfg_.ckpt);
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
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    river_sample_at(env, b5, beta, fresh);
}

void EndgameTrainer::gpu_river_epoch(
    std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>>& envs, int W,
    int ep, std::vector<Sample>& fresh) {
    struct Pending {
        RiverSpec sp;
        std::string sig;
        bool ok = false;
    };
    std::vector<Pending> pend(static_cast<size_t>(cfg_.episodes));
    std::unordered_map<std::string, TreeShape> shapes;
    std::mutex mtx;
    std::atomic<int> next{0};
    auto work = [&](int w) {
        std::mt19937 wrng(static_cast<unsigned>(
            cfg_.seed * 1000003u + ep * 7919u + w * 104729u + 17u));
        while (true) {
            const int e = next.fetch_add(1);
            if (e >= cfg_.episodes) break;
            auto& env = *envs[w];
            bool root = false;
            for (int tries = 0; tries < 50 && !root; ++tries)
                root = sample_street_root(env, wrng, 3);
            if (!root) continue;
            Pending& p = pend[static_cast<size_t>(e)];
            for (int i = 0; i < 5; ++i)
                p.sp.board[i] = static_cast<uint8_t>(env.community_card(i));
            p.sp.r0 = random_range(wrng);
            p.sp.r1 = random_range(wrng);
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
            for (int t = 1; t <= cfg_.t_river; ++t) bs.iterate(t);
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

std::vector<double> EndgameTrainer::random_range(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // Solver-generated leaf ranges are STRUCTURED (sparse, correlated) —
    // iid noise ranges train a net that is useless at solver queries
    // (measured: heldout 0.005 on iid data, 0.15+ on turn-harvest data).
    // DeepStack's fix, used here 70% of the time: recursive stick-breaking —
    // random binary partitions of the combo set with U(0,1) mass splits,
    // yielding realistic sparse/correlated mass patterns.
    if (u(rng) < 0.7) {
        std::vector<double> r(kCombos, 0.0);
        std::vector<int> idx(kCombos);
        for (int i = 0; i < kCombos; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        std::function<void(int, int, double)> split =
            [&](int lo, int hi, double mass) {
            if (hi - lo == 1) {
                r[idx[static_cast<size_t>(lo)]] = mass;
                return;
            }
            const int mid = lo + (hi - lo) / 2;
            const double q = u(rng);
            split(lo, mid, mass * q);
            split(mid, hi, mass * (1.0 - q));
        };
        split(0, kCombos, 1.0);
        return r;
    }
    // iid coverage floor (broad but unstructured)
    const double k = 1.0 + 3.0 * u(rng);
    std::vector<double> r(kCombos);
    for (double& x : r) x = std::pow(u(rng), k);
    return r;
}

void EndgameTrainer::self_play_episode(poker_ppo::PokerEnvironment& env,
                                       std::mt19937& rng,
                                       std::vector<Sample>& fresh) {
    for (int tries = 0; tries < 50; ++tries)
        if (sample_street_root(env, rng, 2)) break;
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
        if (sample_street_root(env, fixed, 2)) break;
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
    std::printf("HUNL endgame ReBeL (%s): T_turn=%d T_river=%d episodes=%d "
                "epochs=%d hidden=%d harvest=%d device=%s workers=%d\n",
                cfg_.river_only ? "direct-river" : "turn-selfplay",
                cfg_.t_turn, cfg_.t_river, cfg_.episodes, cfg_.epochs,
                cfg_.hidden, cfg_.harvest,
                device_.is_cuda() ? "cuda" : "cpu", W);
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
        if (cfg_.river_only && cfg_.gpu_batch > 0 && ep == 1 &&
            !device_.is_cuda())
            std::fprintf(stderr,
                         "  [warn] REBEL_GPU_BATCH on a CPU-torch device is "
                         "~100x SLOWER than the worker path — use it on "
                         "CUDA only\n");
        if (cfg_.river_only && cfg_.gpu_batch > 0) {
            std::vector<Sample> fresh;
            gpu_river_epoch(envs, W, ep, fresh);
            auto t_sp1b = clock::now();
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
            auto t_tr0b = clock::now();
            const double vloss = train_net();
            std::printf("  epoch %3d  replay=%5zu  vloss=%.3e  heldout=%.3e"
                        "  [sp %.0fs train %.0fs]\n",
                        ep, replay_.size(), vloss, heldout,
                        secs(t_sp0, t_sp1b), secs(t_tr0b, clock::now()));
            std::fflush(stdout);
            if (!cfg_.ckpt.empty()) torch::save(net_, cfg_.ckpt);
            continue;
        }
        // episodes fan out over the worker pool (work-stealing counter)
        std::vector<std::vector<Sample>> fresh_w(W);
        std::atomic<int> next{0};
        auto work = [&](int w) {
            std::mt19937 wrng(static_cast<unsigned>(
                cfg_.seed * 1000003u + ep * 7919u + w * 104729u));
            while (true) {
                const int e = next.fetch_add(1);
                if (e >= cfg_.episodes) break;
                if (cfg_.river_only)
                    direct_river_episode(*envs[w], wrng, fresh_w[w]);
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
        if (!cfg_.ckpt.empty()) torch::save(net_, cfg_.ckpt);
    }
}

}  // namespace rebel_hunl
