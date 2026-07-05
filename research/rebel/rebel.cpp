#include "rebel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

namespace rebel {

namespace {
constexpr double kTiny = 1e-12;

// normalize `v` in place with `skip` zeroed; uniform-over-compatible fallback
// when the mass is ~0 (unreached lines — conservative, flagged in rebel.h).
void normalize_belief(std::vector<double>& v, int skip) {
    if (skip >= 0) v[skip] = 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    if (s > kTiny) {
        for (double& x : v) x /= s;
        return;
    }
    int m = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (static_cast<int>(i) != skip) ++m;
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = (static_cast<int>(i) == skip) ? 0.0 : 1.0 / m;
}
}  // namespace

// ─── deck specs ─────────────────────────────────────────────────────────────
DeckSpec kuhn_deck(int n_cards) {
    DeckSpec d;
    d.rank.resize(n_cards);
    for (int i = 0; i < n_cards; ++i) d.rank[i] = i;
    d.has_public = false;
    return d;
}

DeckSpec leduc_deck() {
    DeckSpec d;
    d.rank = {0, 0, 1, 1, 2, 2};
    d.has_public = true;
    return d;
}

PBS PBS::root(const DeckSpec& d) {
    PBS b;
    b.h = "";
    b.pub = -1;
    b.b0.assign(d.n(), 1.0 / d.n());
    b.b1.assign(d.n(), 1.0 / d.n());
    return b;
}

// ─── SubgameSolver ──────────────────────────────────────────────────────────
SubgameSolver::SubgameSolver(const escher::Game& g, const DeckSpec& deck,
                             PBS root, ValueOracle* oracle, bool cfr_avg)
    : g_(g), deck_(deck), root_(std::move(root)), oracle_(oracle),
      cfr_avg_(cfr_avg), n_(deck.n()) {
    build(root_.h);
}

int SubgameSolver::build(const std::string& h) {
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    Node& nd = nodes_.back();
    nd.h = h;
    if (g_.is_terminal(h)) {
        nd.kind = Node::Terminal;
        return id;
    }
    // A '/'-terminated node other than the root is the round boundary: the
    // public card is dealt there — the subgame's depth limit (chance leaf).
    if (!h.empty() && h.back() == '/' && h != root_.h) {
        nd.kind = Node::ChanceLeaf;
        return id;
    }
    nd.kind = Node::Decision;
    nd.player = g_.current_player(h);
    nd.acts = g_.legal_actions(h);
    const int A = static_cast<int>(nd.acts.size());
    nd.regret.assign(static_cast<size_t>(n_) * A, 0.0);
    nd.cum_strat.assign(static_cast<size_t>(n_) * A, 0.0);
    std::vector<int> children;
    children.reserve(A);
    for (int a : nd.acts) {
        const std::string ch = g_.step(h, a);
        children.push_back(build(ch));  // invalidates nd reference
    }
    nodes_[id].child = std::move(children);
    return id;
}

const std::vector<double>& SubgameSolver::terminal_u0(int node) {
    auto it = term_u0_.find(node);
    if (it != term_u0_.end()) return it->second;
    const Node& nd = nodes_[node];
    std::vector<double> u(static_cast<size_t>(n_) * n_, 0.0);
    const int pub_rank = root_.pub >= 0 ? deck_.rank[root_.pub] : 0;
    for (int x = 0; x < n_; ++x)
        for (int y = 0; y < n_; ++y) {
            if (x == y) continue;
            u[static_cast<size_t>(x) * n_ + y] = g_.terminal_util_p0(
                nd.h, {deck_.rank[x], deck_.rank[y], pub_rank});
        }
    return term_u0_.emplace(node, std::move(u)).first->second;
}

std::vector<double> SubgameSolver::current_policy(const Node& nd,
                                                  int card) const {
    const int A = static_cast<int>(nd.acts.size());
    std::vector<double> p(A, 0.0);
    double s = 0.0;
    for (int k = 0; k < A; ++k) {
        const double r = nd.regret[static_cast<size_t>(card) * A + k];
        p[k] = r > 0.0 ? r : 0.0;
        s += p[k];
    }
    if (s > kTiny)
        for (double& x : p) x /= s;
    else
        for (double& x : p) x = 1.0 / A;
    return p;
}

std::vector<double> SubgameSolver::avg_policy(int node, int card) const {
    const Node& nd = nodes_[node];
    const int A = static_cast<int>(nd.acts.size());
    std::vector<double> p(A, 0.0);
    double s = 0.0;
    for (int k = 0; k < A; ++k) {
        p[k] = nd.cum_strat[static_cast<size_t>(card) * A + k];
        s += p[k];
    }
    if (s > kTiny)
        for (double& x : p) x /= s;
    else
        for (double& x : p) x = 1.0 / A;
    return p;
}

void SubgameSolver::avg_reaches(int target, std::vector<double>& r0,
                                std::vector<double>& r1, bool avg) const {
    // forward pass to `target` multiplying policies into the root beliefs
    // (recursion keeps the code obviously correct; trees are tiny).
    r0.assign(n_, 0.0);
    r1.assign(n_, 0.0);
    std::function<bool(int, std::vector<double>&, std::vector<double>&)> rec =
        [&](int i, std::vector<double>& a0, std::vector<double>& a1) -> bool {
        if (i == target) {
            r0 = a0;
            r1 = a1;
            return true;
        }
        const Node& nd = nodes_[i];
        if (nd.kind != Node::Decision) return false;
        const int A = static_cast<int>(nd.acts.size());
        for (int k = 0; k < A; ++k) {
            std::vector<double> c0 = a0, c1 = a1;
            std::vector<double>& mine = nd.player == 0 ? c0 : c1;
            for (int x = 0; x < n_; ++x)
                mine[x] *= avg ? avg_policy(i, x)[k]
                               : current_policy(nd, x)[k];
            if (rec(nd.child[k], c0, c1)) return true;
        }
        return false;
    };
    std::vector<double> a0 = root_.b0, a1 = root_.b1;
    rec(0, a0, a1);
}

void SubgameSolver::refresh_leaves(bool avg) {
    if (oracle_ == nullptr) return;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].kind != Node::ChanceLeaf) continue;
        std::vector<double> r0, r1;
        avg_reaches(static_cast<int>(i), r0, r1, avg);
        auto& per_pub = leaf_v_[static_cast<int>(i)];
        per_pub.resize(n_);
        for (int c = 0; c < n_; ++c) {
            PBS beta;
            beta.h = nodes_[i].h;
            beta.pub = c;
            beta.b0 = r0;
            beta.b1 = r1;
            normalize_belief(beta.b0, c);
            normalize_belief(beta.b1, c);
            oracle_->value(beta, per_pub[c]);
        }
    }
}

std::vector<double> SubgameSolver::walk(int i, int upd, int t,
                                        std::vector<double>& my_reach,
                                        std::vector<double>& opp_reach) {
    Node& nd = nodes_[i];
    std::vector<double> cfv(n_, 0.0);

    if (nd.kind == Node::Terminal) {
        const auto& u = terminal_u0(i);
        for (int x = 0; x < n_; ++x) {
            double v = 0.0;
            for (int y = 0; y < n_; ++y) {
                if (y == x) continue;
                // u is [p0 card][p1 card] in P0 frame.
                v += upd == 0 ? opp_reach[y] * u[static_cast<size_t>(x) * n_ + y]
                              : opp_reach[y] * -u[static_cast<size_t>(y) * n_ + x];
            }
            cfv[x] = v;
        }
        return cfv;
    }

    if (nd.kind == Node::ChanceLeaf) {
        // v̂ queried at CFR-AVG beliefs (refresh_leaves); chance is uniform
        // over the n-2 cards outside both players' hands, so the per-(x,y)
        // outcome weight is 1/(n-2) and Σ_c over compatible c is exactly 1.
        const auto& per_pub = leaf_v_.at(i);
        const double chance = 1.0 / (n_ - 2);
        double osum = 0.0;
        for (int y = 0; y < n_; ++y) osum += opp_reach[y];
        for (int c = 0; c < n_; ++c) {
            const auto& v = per_pub[c][upd];
            if (v.empty()) continue;
            for (int x = 0; x < n_; ++x) {
                if (x == c) continue;
                const double mass = osum - opp_reach[x] - opp_reach[c];
                if (mass <= 0.0) continue;
                cfv[x] += chance * mass * v[x];
            }
        }
        return cfv;
    }

    const int A = static_cast<int>(nd.acts.size());
    if (nd.player == upd) {
        std::vector<std::vector<double>> sig(n_);
        for (int x = 0; x < n_; ++x) sig[x] = current_policy(nd, x);
        std::vector<std::vector<double>> cfv_a(A);
        for (int k = 0; k < A; ++k) {
            std::vector<double> child_reach(n_);
            for (int x = 0; x < n_; ++x)
                child_reach[x] = my_reach[x] * sig[x][k];
            cfv_a[k] = walk(nd.child[k], upd, t, child_reach, opp_reach);
        }
        for (int x = 0; x < n_; ++x) {
            double v = 0.0;
            for (int k = 0; k < A; ++k) v += sig[x][k] * cfv_a[k][x];
            cfv[x] = v;
            for (int k = 0; k < A; ++k) {
                double& r = nd.regret[static_cast<size_t>(x) * A + k];
                r += cfv_a[k][x] - v;
                if (r < 0.0) r = 0.0;  // RM⁺
                nd.cum_strat[static_cast<size_t>(x) * A + k] +=
                    static_cast<double>(t) * my_reach[x] * sig[x][k];
            }
        }
        return cfv;
    }

    // opponent node: fold σ into the opponent reach, sum child values
    for (int k = 0; k < A; ++k) {
        std::vector<double> child_opp(n_);
        for (int x = 0; x < n_; ++x)
            child_opp[x] = opp_reach[x] * current_policy(nd, x)[k];
        auto cv = walk(nd.child[k], upd, t, my_reach, child_opp);
        for (int x = 0; x < n_; ++x) cfv[x] += cv[x];
    }
    return cfv;
}

void SubgameSolver::iterate(int t) {
    refresh_leaves(cfr_avg_);
    for (int upd = 0; upd < 2; ++upd) {
        std::vector<double> mine = upd == 0 ? root_.b0 : root_.b1;
        std::vector<double> opp = upd == 0 ? root_.b1 : root_.b0;
        walk(0, upd, t, mine, opp);
    }
}

void SubgameSolver::root_values(std::array<std::vector<double>, 2>& v,
                                std::array<std::vector<double>, 2>& mask) {
    // one pass under the FINAL average profile for each player (no updates):
    // temporarily read policies from cum_strat via a local recursion.
    std::function<std::vector<double>(int, std::vector<double>&)> rec_for;
    int upd = 0;
    std::function<std::vector<double>(int, std::vector<double>&)> rec =
        [&](int i, std::vector<double>& opp_reach) -> std::vector<double> {
        Node& nd = nodes_[i];
        std::vector<double> cfv(n_, 0.0);
        if (nd.kind == Node::Terminal) {
            const auto& u = terminal_u0(i);
            for (int x = 0; x < n_; ++x)
                for (int y = 0; y < n_; ++y) {
                    if (y == x) continue;
                    cfv[x] += upd == 0
                        ? opp_reach[y] * u[static_cast<size_t>(x) * n_ + y]
                        : opp_reach[y] * -u[static_cast<size_t>(y) * n_ + x];
                }
            return cfv;
        }
        if (nd.kind == Node::ChanceLeaf) {
            const auto& per_pub = leaf_v_.at(i);
            const double chance = 1.0 / (n_ - 2);
            double osum = 0.0;
            for (int y = 0; y < n_; ++y) osum += opp_reach[y];
            for (int c = 0; c < n_; ++c) {
                const auto& lv = per_pub[c][upd];
                if (lv.empty()) continue;
                for (int x = 0; x < n_; ++x) {
                    if (x == c) continue;
                    const double mass = osum - opp_reach[x] - opp_reach[c];
                    if (mass > 0.0) cfv[x] += chance * mass * lv[x];
                }
            }
            return cfv;
        }
        const int A = static_cast<int>(nd.acts.size());
        for (int k = 0; k < A; ++k) {
            if (nd.player != upd) {
                std::vector<double> child_opp(n_);
                for (int x = 0; x < n_; ++x)
                    child_opp[x] = opp_reach[x] * avg_policy(i, x)[k];
                auto cv = rec(nd.child[k], child_opp);
                for (int x = 0; x < n_; ++x) cfv[x] += cv[x];
            } else {
                auto cv = rec(nd.child[k], opp_reach);
                for (int x = 0; x < n_; ++x)
                    cfv[x] += avg_policy(i, x)[k] * cv[x];
            }
        }
        return cfv;
    };

    // Targets are values OF the final average profile → leaf beliefs from
    // the average regardless of the solve's leaf-belief mode.
    refresh_leaves(true);
    for (upd = 0; upd < 2; ++upd) {
        std::vector<double> opp = upd == 0 ? root_.b1 : root_.b0;
        auto cfv = rec(0, opp);
        const auto& own = upd == 0 ? root_.b0 : root_.b1;
        v[upd].assign(n_, 0.0);
        mask[upd].assign(n_, 0.0);
        double osum = 0.0;
        for (double y : opp) osum += y;
        for (int x = 0; x < n_; ++x) {
            if (x == root_.pub) continue;
            const double m = osum - opp[x];
            if (own[x] > kTiny && m > kTiny) {
                v[upd][x] = cfv[x] / m;
                mask[upd][x] = 1.0;
            }
        }
    }
}

bool SubgameSolver::sample_leaf(std::mt19937& rng, double eps, int explorer,
                                PBS* out) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    // sample compatible private hands from the root beliefs
    std::vector<double> w(static_cast<size_t>(n_) * n_, 0.0);
    double ws = 0.0;
    for (int x = 0; x < n_; ++x)
        for (int y = 0; y < n_; ++y)
            if (x != y) {
                w[static_cast<size_t>(x) * n_ + y] = root_.b0[x] * root_.b1[y];
                ws += w[static_cast<size_t>(x) * n_ + y];
            }
    int hand[2] = {0, 1};
    if (ws > kTiny) {
        double r = u01(rng) * ws, acc = 0.0;
        for (int x = 0; x < n_ && r >= 0; ++x)
            for (int y = 0; y < n_; ++y) {
                acc += w[static_cast<size_t>(x) * n_ + y];
                if (r <= acc) {
                    hand[0] = x;
                    hand[1] = y;
                    r = -1;
                    break;
                }
            }
    }

    int i = 0;
    std::vector<double> p0 = root_.b0, p1 = root_.b1;  // walked posteriors
    while (true) {
        const Node& nd = nodes_[i];
        if (nd.kind == Node::Terminal) return false;    // episode ends
        if (nd.kind == Node::ChanceLeaf) {
            std::vector<int> avail;
            for (int c = 0; c < n_; ++c)
                if (c != hand[0] && c != hand[1]) avail.push_back(c);
            const int c = avail[rng() % avail.size()];
            out->h = nd.h;
            out->pub = c;
            out->b0 = p0;
            out->b1 = p1;
            normalize_belief(out->b0, c);
            normalize_belief(out->b1, c);
            return true;
        }
        const int A = static_cast<int>(nd.acts.size());
        const int me = nd.player;
        auto pol = avg_policy(i, hand[me]);   // CFR-AVG pairing (Alg 1)
        int k = 0;
        if (me == explorer && u01(rng) < eps) {
            k = static_cast<int>(rng() % A);
        } else {
            double r = u01(rng), acc = 0.0;
            for (int j = 0; j < A; ++j) {
                acc += pol[j];
                if (r <= acc) {
                    k = j;
                    break;
                }
                k = j;
            }
        }
        // public Bayes update of the ACTING player's belief for the action
        std::vector<double>& mine = me == 0 ? p0 : p1;
        for (int x = 0; x < n_; ++x) mine[x] *= avg_policy(i, x)[k];
        i = nd.child[k];
    }
}

// ─── value net & featurizer ─────────────────────────────────────────────────
ValueNetImpl::ValueNetImpl(int d_in, int d_out, int hidden) {
    l1 = register_module("l1", torch::nn::Linear(d_in, hidden));
    l2 = register_module("l2", torch::nn::Linear(hidden, hidden));
    l3 = register_module("l3", torch::nn::Linear(hidden, d_out));
}
torch::Tensor ValueNetImpl::forward(torch::Tensor x) {
    x = torch::relu(l1(x));
    x = torch::relu(l2(x));
    return l3(x);
}

Featurizer::Featurizer(const escher::Game& g, const DeckSpec& deck)
    : n_(deck.n()) {
    // enumerate '/'-terminated public histories (round-2 roots = the only
    // PBSs the net is ever queried or trained at in a 2-round game)
    std::function<void(const std::string&)> rec = [&](const std::string& h) {
        if (g.is_terminal(h)) return;
        if (!h.empty() && h.back() == '/') {
            if (!hidx_.count(h)) hidx_.emplace(h, static_cast<int>(hidx_.size()));
            return;
        }
        for (int a : g.legal_actions(h)) rec(g.step(h, a));
    };
    rec("");
    dim_ = static_cast<int>(hidx_.size()) + n_ + 2 * n_;
    if (dim_ == 3 * n_) dim_ += 1;  // no public round (Kuhn): keep dim valid
}

std::vector<float> Featurizer::operator()(const PBS& beta) const {
    std::vector<float> f(dim_, 0.0f);
    auto it = hidx_.find(beta.h);
    if (it != hidx_.end()) f[it->second] = 1.0f;
    const int off = static_cast<int>(hidx_.size());
    if (beta.pub >= 0) f[off + beta.pub] = 1.0f;
    for (int x = 0; x < n_; ++x) {
        f[off + n_ + x] = static_cast<float>(beta.b0[x]);
        f[off + 2 * n_ + x] = static_cast<float>(beta.b1[x]);
    }
    return f;
}

void NetOracle::value(const PBS& beta,
                      std::array<std::vector<double>, 2>& out) {
    torch::NoGradGuard ng;
    auto f = feat_(beta);
    auto x = torch::from_blob(f.data(), {1, static_cast<long>(f.size())},
                              torch::kFloat)
                 .clone();
    auto y = net_->forward(x).squeeze(0);
    const int n = static_cast<int>(beta.b0.size());
    auto acc = y.accessor<float, 1>();
    for (int p = 0; p < 2; ++p) {
        out[p].assign(n, 0.0);
        for (int c = 0; c < n; ++c) out[p][c] = acc[p * n + c];
    }
}

void ExactOracle::value(const PBS& beta,
                        std::array<std::vector<double>, 2>& out) {
    SubgameSolver s(g_, deck_, beta, nullptr);  // final round: no leaves
    for (int t = 1; t <= iters_; ++t) s.iterate(t);
    std::array<std::vector<double>, 2> mask;
    s.root_values(out, mask);
}

// ─── Trainer (Algorithm 1) ──────────────────────────────────────────────────
Trainer::Trainer(const escher::Game& g, DeckSpec deck, Config cfg)
    : g_(g), deck_(std::move(deck)), cfg_(cfg), feat_(g, deck_),
      rng_(static_cast<unsigned>(cfg.seed)) {
    torch::manual_seed(cfg_.seed);
    net_ = ValueNet(feat_.dim(), 2 * deck_.n(), cfg_.hidden);
    opt_ = std::make_unique<torch::optim::Adam>(net_->parameters(), cfg_.lr);
}

void Trainer::push_targets(SubgameSolver& s) {
    const PBS& beta = s.root();
    // the net is only queried at round-boundary PBSs — train it there only
    if (beta.pub < 0) return;
    std::array<std::vector<double>, 2> v, m;
    s.root_values(v, m);
    Sample smp;
    smp.beta = beta;
    smp.feat = feat_(beta);
    const int n = deck_.n();
    smp.target.assign(2 * n, 0.0f);
    smp.mask.assign(2 * n, 0.0f);
    for (int p = 0; p < 2; ++p)
        for (int x = 0; x < n; ++x) {
            smp.target[p * n + x] = static_cast<float>(v[p][x]);
            smp.mask[p * n + x] = static_cast<float>(m[p][x]);
        }
    ++replay_seen_;
    if (static_cast<int>(replay_.size()) < cfg_.replay_cap) {
        replay_.push_back(std::move(smp));
    } else {
        std::uniform_int_distribution<long> d(0, replay_seen_ - 1);
        const long j = d(rng_);
        if (j < static_cast<long>(replay_.size()))
            replay_[static_cast<size_t>(j)] = std::move(smp);
    }
}

void Trainer::self_play_episode(std::mt19937& rng) {
    NetOracle oracle(net_, feat_);
    PBS beta = PBS::root(deck_);
    const int explorer = static_cast<int>(rng() & 1);
    while (!g_.is_terminal(beta.h)) {
        SubgameSolver s(g_, deck_, beta, &oracle, cfg_.cfr_avg);
        std::uniform_int_distribution<int> dt(1, cfg_.t_train);
        const int tstar = dt(rng);
        PBS next;
        bool have_next = false, next_terminal = false;
        for (int t = 1; t <= cfg_.t_train; ++t) {
            s.iterate(t);
            if (t == tstar)
                next_terminal =
                    !(have_next = s.sample_leaf(rng, cfg_.eps_explore,
                                                explorer, &next));
        }
        push_targets(s);
        if (!have_next || next_terminal) break;
        beta = std::move(next);
    }
}

double Trainer::train_value_net() {
    if (replay_.empty()) return 0.0;
    const int n_out = 2 * deck_.n();
    double last = 0.0;
    std::uniform_int_distribution<size_t> pick(0, replay_.size() - 1);
    for (int step = 0; step < cfg_.sgd_steps; ++step) {
        const int B =
            std::min<int>(cfg_.batch, static_cast<int>(replay_.size()));
        auto X = torch::empty({B, feat_.dim()}, torch::kFloat);
        auto Y = torch::empty({B, n_out}, torch::kFloat);
        auto M = torch::empty({B, n_out}, torch::kFloat);
        for (int b = 0; b < B; ++b) {
            const Sample& s = replay_[pick(rng_)];
            std::copy(s.feat.begin(), s.feat.end(),
                      X.data_ptr<float>() + static_cast<size_t>(b) * feat_.dim());
            std::copy(s.target.begin(), s.target.end(),
                      Y.data_ptr<float>() + static_cast<size_t>(b) * n_out);
            std::copy(s.mask.begin(), s.mask.end(),
                      M.data_ptr<float>() + static_cast<size_t>(b) * n_out);
        }
        opt_->zero_grad();
        auto pred = net_->forward(X);
        auto loss = ((pred - Y).pow(2) * M).sum() / M.sum().clamp_min(1.0);
        loss.backward();
        opt_->step();
        if (step == cfg_.sgd_steps - 1) last = loss.item<double>();
    }
    return last;
}

double Trainer::value_probe_mse(int k) {
    if (replay_.empty()) return 0.0;
    ExactOracle exact(g_, deck_, cfg_.exact_iters);
    NetOracle net(net_, feat_);
    std::uniform_int_distribution<size_t> pick(0, replay_.size() - 1);
    double se = 0.0, cnt = 0.0;
    for (int i = 0; i < k; ++i) {
        const Sample& s = replay_[pick(rng_)];
        std::array<std::vector<double>, 2> ve, vn;
        exact.value(s.beta, ve);
        net.value(s.beta, vn);
        const int n = deck_.n();
        for (int p = 0; p < 2; ++p)
            for (int x = 0; x < n; ++x)
                if (s.mask[p * n + x] > 0.5f) {
                    const double e = ve[p][x] - vn[p][x];
                    se += e * e;
                    cnt += 1.0;
                }
    }
    return cnt > 0 ? se / cnt : 0.0;
}

escher::Strategy Trainer::agent_strategy(ValueOracle* oracle, int t_solve) {
    escher::Strategy S;
    std::function<void(const PBS&)> rec = [&](const PBS& beta) {
        SubgameSolver s(g_, deck_, beta, oracle, cfg_.cfr_avg);
        for (int t = 1; t <= t_solve; ++t) s.iterate(t);
        const int pub_rank = beta.pub >= 0 ? deck_.rank[beta.pub] : 0;
        const auto& nodes = s.nodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& nd = nodes[i];
            if (nd.kind != SubgameSolver::Node::Decision) continue;
            for (int x = 0; x < deck_.n(); ++x) {
                if (x == beta.pub) continue;
                escher::Cards c{0, 0, pub_rank};
                c[nd.player] = deck_.rank[x];
                const std::string key = g_.infoset_key(nd.h, c);
                auto pol = s.avg_policy(static_cast<int>(i), x);
                std::vector<double> full(g_.n_actions(), 0.0);
                for (size_t k = 0; k < nd.acts.size(); ++k)
                    full[nd.acts[k]] = pol[k];
                S[key] = std::move(full);
            }
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].kind != SubgameSolver::Node::ChanceLeaf) continue;
            std::vector<double> r0, r1;
            s.avg_reaches(static_cast<int>(i), r0, r1);
            for (int c = 0; c < deck_.n(); ++c) {
                PBS nb;
                nb.h = nodes[i].h;
                nb.pub = c;
                nb.b0 = r0;
                nb.b1 = r1;
                if (cfg_.eval_smooth > 0.0) {  // unreached-gap diagnostic
                    for (int x = 0; x < deck_.n(); ++x) {
                        nb.b0[x] += cfg_.eval_smooth;
                        nb.b1[x] += cfg_.eval_smooth;
                    }
                }
                normalize_belief(nb.b0, c);
                normalize_belief(nb.b1, c);
                rec(nb);
            }
        }
    };
    rec(PBS::root(deck_));
    return S;
}

void Trainer::run() {
    std::printf("ReBeL on %s: T=%d, %d episodes/epoch, %d epochs, "
                "eps=%.2f, hidden=%d\n",
                g_.name().c_str(), cfg_.t_train, cfg_.episodes, cfg_.epochs,
                cfg_.eps_explore, cfg_.hidden);
    NetOracle oracle(net_, feat_);
    for (int ep = 1; ep <= cfg_.epochs; ++ep) {
        for (int e = 0; e < cfg_.episodes; ++e) self_play_episode(rng_);
        const double vloss = train_value_net();
        const double probe = value_probe_mse(32);
        auto strat = agent_strategy(&oracle, cfg_.t_eval);
        const double expl = escher::exploitability(g_, strat);
        std::printf("  epoch %3d  replay=%6zu  vloss=%.5f  "
                    "probe-mse=%.5f  expl(agent)=%.5f\n",
                    ep, replay_.size(), vloss, probe, expl);
        std::fflush(stdout);
    }
}

}  // namespace rebel
