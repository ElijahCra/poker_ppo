#include "escher.h"

#include <cmath>
#include <cstdio>

namespace escher {

// ─── Featurizer ─────────────────────────────────────────────────────────────
static std::string hist_of(const std::string& key) {
    auto pos = key.rfind('|');
    return pos == std::string::npos ? key : key.substr(pos + 1);
}

Featurizer::Featurizer(const Game& g, int n_ranks)
    : g_(g), n_ranks_(n_ranks) {
    std::set<std::string> keys;
    std::set<Cards> seen;
    std::function<void(const std::string&, const Cards&)> rec =
        [&](const std::string& h, const Cards& c) {
            if (g_.is_terminal(h)) return;
            keys.insert(g_.infoset_key(h, c));
            for (int a : g_.legal_actions(h)) rec(g_.step(h, a), c);
        };
    for (const auto& d : g_.deals()) {
        if (seen.count(d.cards)) continue;
        seen.insert(d.cards);
        rec("", d.cards);
    }
    std::set<std::string> hists;
    has_public_ = false;
    for (const auto& k : keys) {
        std::string hh = hist_of(k);
        hists.insert(hh);
        if (hh.find('/') != std::string::npos) has_public_ = true;
    }
    int i = 0;
    for (const auto& h : hists) hidx_[h] = i++;
    n_hist_ = static_cast<int>(hists.size());
    pub_dim_ = has_public_ ? (n_ranks_ + 1) : 0;
    inf_dim_ = 1 + n_ranks_ + pub_dim_ + n_hist_ + 1;
    extra_dim_ = 3 + (has_public_ ? 2 : 0);
    full_dim_ = 2 * n_ranks_ + pub_dim_ + n_hist_ + extra_dim_;
}

std::vector<float> Featurizer::infoset(const std::string& h, const Cards& c,
                                       int player) const {
    std::vector<float> v(inf_dim_, 0.0f);
    int i = 0, card = c[player];
    v[i] = static_cast<float>(card) / std::max(1, n_ranks_ - 1);
    i += 1;
    v[i + card] = 1.0f; i += n_ranks_;
    if (has_public_) { v[i + public_idx(h, c)] = 1.0f; i += pub_dim_; }
    v[i + hidx_.at(h)] = 1.0f; i += n_hist_;
    v[i] = static_cast<float>(player);
    return v;
}

std::vector<float> Featurizer::full(const std::string& h, const Cards& c) const {
    // PRIVILEGED: always encodes the public card (value net conditions on the
    // whole deal). Plus showdown interaction features.
    std::vector<float> v(full_dim_, 0.0f);
    int i = 0;
    v[i + c[0]] = 1.0f; i += n_ranks_;
    v[i + c[1]] = 1.0f; i += n_ranks_;
    if (has_public_) { v[i + 1 + c[2]] = 1.0f; i += pub_dim_; }  // always revealed
    v[i + hidx_.at(h)] = 1.0f; i += n_hist_;
    int r0, r1;
    if (has_public_) {
        int pub = c[2];
        bool p0p = (c[0] == pub), p1p = (c[1] == pub);
        v[i] = p0p ? 1.0f : 0.0f; v[i + 1] = p1p ? 1.0f : 0.0f; i += 2;
        r0 = p0p ? n_ranks_ + c[0] : c[0];
        r1 = p1p ? n_ranks_ + c[1] : c[1];
    } else { r0 = c[0]; r1 = c[1]; }
    v[i + (r0 > r1 ? 0 : (r0 == r1 ? 1 : 2))] = 1.0f;  // win/tie/lose
    return v;
}

// ─── NeuralESCHER ───────────────────────────────────────────────────────────
NeuralESCHER::NeuralESCHER(const Game& g, int n_ranks, Config cfg)
    : g_(g), feat_(g, n_ranks), cfg_(cfg), A_(g.n_actions()), rng_(cfg.seed),
      reg_buf_(200000, cfg.seed + 1), strat_buf_(200000, cfg.seed + 2),
      val_buf_(200000, cfg.seed + 3) {
    torch::manual_seed(cfg.seed);
    regret_net_ = MLP(feat_.inf_dim(), A_, cfg.hidden);
    avg_net_ = MLP(feat_.inf_dim(), A_, cfg.hidden);
    if (cfg.mode == Mode::Net) {
        value_net_ = MLP(feat_.full_dim(), 1, cfg.val_hidden);
        value_opt_ = std::make_shared<torch::optim::Adam>(
            value_net_->parameters(), torch::optim::AdamOptions(1e-3));
        v_scale_ = compute_v_scale();
    }
}

double NeuralESCHER::compute_v_scale() {
    double mx = 1.0;
    std::set<Cards> seen;
    std::function<void(const std::string&, const Cards&)> rec =
        [&](const std::string& h, const Cards& c) {
            if (g_.is_terminal(h)) { mx = std::max(mx, std::abs(g_.terminal_util_p0(h, c))); return; }
            for (int a : g_.legal_actions(h)) rec(g_.step(h, a), c);
        };
    for (const auto& d : g_.deals()) {
        if (seen.count(d.cards)) continue;
        seen.insert(d.cards);
        rec("", d.cards);
    }
    return mx;
}

std::vector<double> NeuralESCHER::sigma(const std::string& h, const Cards& c,
                                        int player, const std::vector<int>& legal) {
    auto f = feat_.infoset(h, c, player);
    torch::Tensor r;
    { torch::NoGradGuard ng; r = regret_net_->forward(feat_tensor(f)).squeeze(0); }
    std::vector<double> probs(A_, 0.0);
    double s = 0.0;
    for (int a : legal) { double v = r[a].item<double>(); if (v > 0) { probs[a] = v; s += v; } }
    if (s > 1e-12) for (int a : legal) probs[a] /= s;
    else for (int a : legal) probs[a] = 1.0 / legal.size();
    return probs;
}

double NeuralESCHER::exact_value(const std::string& h, const Cards& c) {
    if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
    int player = g_.current_player(h);
    auto legal = g_.legal_actions(h);
    auto sig = sigma(h, c, player, legal);
    double v = 0.0;
    for (int a : legal) v += sig[a] * exact_value(g_.step(h, a), c);
    return v;
}

double NeuralESCHER::net_value(const std::string& h, const Cards& c) {
    if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
    auto f = feat_.full(h, c);
    torch::Tensor o;
    { torch::NoGradGuard ng; o = value_net_->forward(feat_tensor(f)); }
    return o.item<double>() * v_scale_;
}

void NeuralESCHER::collect_regret(long t) {
    std::unordered_map<std::string, std::vector<double>> inst, strat;
    struct Meta { std::string h; Cards c; int player; std::vector<int> legal; };
    std::unordered_map<std::string, Meta> meta;

    std::function<double(const std::string&, const Cards&, double, double, double)> rec =
        [&](const std::string& h, const Cards& c, double p0, double p1, double q) -> double {
            if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
            int player = g_.current_player(h);
            auto legal = g_.legal_actions(h);
            std::string I = g_.infoset_key(h, c);
            auto sig = sigma(h, c, player, legal);
            std::vector<double> util_a(A_, 0.0);
            double node = 0.0;
            for (int a : legal) {
                util_a[a] = rec(g_.step(h, a), c,
                                player == 0 ? p0 * sig[a] : p0,
                                player == 1 ? p1 * sig[a] : p1, q);
                node += sig[a] * util_a[a];
            }
            double cf = q * (player == 0 ? p1 : p0);
            double own = q * (player == 0 ? p0 : p1);
            double sign = player == 0 ? 1.0 : -1.0;
            if (!inst.count(I)) inst[I] = std::vector<double>(A_, 0.0);
            for (int a : legal) {
                double va, vn;
                if (cfg_.mode == Mode::Net) { va = value_of(g_.step(h, a), c); vn = value_of(h, c); }
                else { va = util_a[a]; vn = node; }
                inst[I][a] += cf * sign * (va - vn);
            }
            if (!strat.count(I)) strat[I] = std::vector<double>(A_, 0.0);
            for (int a : legal) strat[I][a] += own * sig[a];
            meta[I] = {h, c, player, legal};
            return node;
        };

    for (const auto& d : g_.deals()) rec("", d.cards, 1.0, 1.0, d.prob);

    for (auto& kv : inst) {
        const auto& m = meta[kv.first];
        auto f = feat_.infoset(m.h, m.c, m.player);
        std::vector<float> rvec(A_);
        for (int a = 0; a < A_; ++a) rvec[a] = static_cast<float>(kv.second[a]);
        reg_buf_.add({f, rvec, static_cast<float>(t)});
    }
    for (auto& kv : strat) {
        const auto& m = meta[kv.first];
        double tot = 0.0;
        for (double x : kv.second) tot += x;
        if (tot <= 0) continue;
        auto f = feat_.infoset(m.h, m.c, m.player);
        std::vector<float> probs(A_);
        for (int a = 0; a < A_; ++a) probs[a] = static_cast<float>(kv.second[a] / tot);
        strat_buf_.add({f, probs, static_cast<float>(t)});
        if (!tab_ss_.count(kv.first)) tab_ss_[kv.first] = std::vector<double>(A_, 0.0);
        for (int a = 0; a < A_; ++a) tab_ss_[kv.first][a] += t * (kv.second[a] / tot);
        tab_w_[kv.first] += t;
        tab_legal_[kv.first] = m.legal;
    }
}

// fresh-init regret net + global-normalised weighted-MSE fit (Deep CFR)
void NeuralESCHER::fit_regret() {
    regret_net_ = MLP(feat_.inf_dim(), A_, cfg_.hidden);
    auto& data = reg_buf_.data();
    if (data.empty()) return;
    torch::optim::Adam opt(regret_net_->parameters(), torch::optim::AdamOptions(3e-3));
    // global target scale (RM is scale-invariant)
    double ss = 0.0; long cnt = 0;
    for (auto& s : data) for (float x : s.target) { ss += static_cast<double>(x) * x; ++cnt; }
    double scale = std::max(1e-6, std::sqrt(ss / std::max(1L, cnt)));
    int D = feat_.inf_dim(), mb = 512;
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    for (int step = 0; step < cfg_.reg_steps; ++step) {
        int B = std::min<int>(mb, data.size());
        std::vector<float> xb(B * D), yb(B * A_), wb(B);
        for (int b = 0; b < B; ++b) {
            size_t idx = pick(rng_);
            const auto& s = data[idx];
            std::copy(s.feat.begin(), s.feat.end(), xb.begin() + b * D);
            for (int a = 0; a < A_; ++a) yb[b * A_ + a] = s.target[a] / scale;
            wb[b] = s.w;
        }
        auto X = torch::from_blob(xb.data(), {B, D}, torch::kFloat).clone();
        auto Y = torch::from_blob(yb.data(), {B, A_}, torch::kFloat).clone();
        auto W = torch::from_blob(wb.data(), {B, 1}, torch::kFloat).clone();
        opt.zero_grad();
        auto pred = regret_net_->forward(X);
        auto loss = (W * (pred - Y).pow(2)).mean();
        loss.backward();
        opt.step();
    }
}

// fresh-init avg net + iteration-weighted cross-entropy to the avg σ
void NeuralESCHER::fit_avg() {
    avg_net_ = MLP(feat_.inf_dim(), A_, cfg_.hidden);
    auto& data = strat_buf_.data();
    if (data.empty()) return;
    torch::optim::Adam opt(avg_net_->parameters(), torch::optim::AdamOptions(2e-3));
    int D = feat_.inf_dim(), mb = 1024, steps = 600;
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    for (int step = 0; step < steps; ++step) {
        int B = std::min<int>(mb, data.size());
        std::vector<float> xb(B * D), yb(B * A_), wb(B);
        for (int b = 0; b < B; ++b) {
            size_t idx = pick(rng_);
            const auto& s = data[idx];
            std::copy(s.feat.begin(), s.feat.end(), xb.begin() + b * D);
            for (int a = 0; a < A_; ++a) yb[b * A_ + a] = s.target[a];
            wb[b] = s.w;
        }
        auto X = torch::from_blob(xb.data(), {B, D}, torch::kFloat).clone();
        auto Y = torch::from_blob(yb.data(), {B, A_}, torch::kFloat).clone();
        auto W = torch::from_blob(wb.data(), {B}, torch::kFloat).clone();
        opt.zero_grad();
        auto logp = torch::log_softmax(avg_net_->forward(X), 1);
        auto loss = (W * (-(Y * logp).sum(1))).mean();
        loss.backward();
        opt.step();
    }
}

int NeuralESCHER::choose_sigma(const std::vector<int>& legal, const std::vector<double>& sig) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_), acc = 0.0;
    for (int a : legal) { acc += sig[a]; if (r <= acc) return a; }
    return legal.back();
}

Cards NeuralESCHER::sample_deal() {
    auto deals = g_.deals();
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_), acc = 0.0;
    for (auto& d : deals) { acc += d.prob; if (r <= acc) return d.cards; }
    return deals.back().cards;
}

// frozen-target fitted value iteration (see header)
void NeuralESCHER::train_value() {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int tr = 0; tr < 32; ++tr) {  // grow coverage (expl-mixed sampling)
        Cards c = sample_deal();
        std::string h;
        while (!g_.is_terminal(h)) {
            val_buf_.add({h, c});
            auto legal = g_.legal_actions(h);
            int a;
            if (u(rng_) < v_expl_) {
                std::uniform_int_distribution<size_t> pk(0, legal.size() - 1);
                a = legal[pk(rng_)];
            } else {
                int p = g_.current_player(h);
                a = choose_sigma(legal, sigma(h, c, p, legal));
            }
            h = g_.step(h, a);
        }
    }
    auto& data = val_buf_.data();
    if (data.empty()) return;
    // snapshot target net
    MLP target(feat_.full_dim(), 1, cfg_.val_hidden);
    { torch::NoGradGuard ng;
      auto src = value_net_->parameters(); auto dst = target->parameters();
      for (size_t i = 0; i < src.size(); ++i) dst[i].copy_(src[i]); }
    auto tval = [&](const std::string& h, const Cards& c) -> double {
        if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
        auto f = feat_.full(h, c);
        torch::Tensor o; { torch::NoGradGuard ng; o = target->forward(feat_tensor(f)); }
        return o.item<double>() * v_scale_;
    };
    int n_states = std::min<int>(2048, data.size());
    int D = feat_.full_dim();
    std::vector<float> xb(n_states * D), yb(n_states);
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    for (int b = 0; b < n_states; ++b) {
        const auto& st = data[pick(rng_)];
        auto legal = g_.legal_actions(st.h);
        int p = g_.current_player(st.h);
        auto sig = sigma(st.h, st.c, p, legal);
        double tgt = 0.0;
        for (int a : legal) tgt += sig[a] * tval(g_.step(st.h, a), st.c);
        auto f = feat_.full(st.h, st.c);
        std::copy(f.begin(), f.end(), xb.begin() + b * D);
        yb[b] = static_cast<float>(tgt / v_scale_);
    }
    auto X = torch::from_blob(xb.data(), {n_states, D}, torch::kFloat).clone();
    auto Y = torch::from_blob(yb.data(), {n_states, 1}, torch::kFloat).clone();
    for (int step = 0; step < 120; ++step) {
        value_opt_->zero_grad();
        auto loss = torch::mse_loss(value_net_->forward(X), Y);
        loss.backward();
        value_opt_->step();
    }
}

Strategy NeuralESCHER::tabular_average() {
    Strategy out;
    std::set<Cards> seen;
    std::function<void(const std::string&, const Cards&)> rec =
        [&](const std::string& h, const Cards& c) {
            if (g_.is_terminal(h)) return;
            std::string I = g_.infoset_key(h, c);
            auto legal = g_.legal_actions(h);
            if (!out.count(I)) {
                std::vector<double> p(A_, 0.0);
                auto it = tab_w_.find(I);
                if (it != tab_w_.end() && it->second > 0) {
                    double w = it->second;
                    for (int a : legal) p[a] = tab_ss_[I][a] / w;
                } else for (int a : legal) p[a] = 1.0 / legal.size();
                out[I] = p;
            }
            for (int a : legal) rec(g_.step(h, a), c);
        };
    for (const auto& d : g_.deals()) {
        if (seen.count(d.cards)) continue;
        seen.insert(d.cards);
        rec("", d.cards);
    }
    return out;
}

std::pair<double, double> NeuralESCHER::value_diag() {
    double se = 0.0, w = 0.0, mx = 0.0;
    std::function<double(const std::string&, const Cards&, double)> rec =
        [&](const std::string& h, const Cards& c, double reach) -> double {
            if (g_.is_terminal(h)) return g_.terminal_util_p0(h, c);
            int player = g_.current_player(h);
            auto legal = g_.legal_actions(h);
            auto sig = sigma(h, c, player, legal);
            double ev = 0.0;
            for (int a : legal) ev += sig[a] * rec(g_.step(h, a), c, reach * sig[a]);
            double nv = net_value(h, c);
            se += reach * (nv - ev) * (nv - ev);
            w += reach;
            mx = std::max(mx, std::abs(nv - ev));
            return ev;
        };
    for (const auto& d : g_.deals()) rec("", d.cards, d.prob);
    return {std::sqrt(se / std::max(w, 1e-9)), mx};
}

void NeuralESCHER::run(int iters, int eval_every) {
    long target = t_ + iters;
    while (t_ < target) {
        ++t_;
        std::pair<double, double> vd{0, 0};
        bool eval_now = (t_ % eval_every == 0 || t_ == target);
        if (cfg_.mode == Mode::Net) {
            for (int s = 0; s < cfg_.val_sweeps; ++s) train_value();
            if (eval_now) vd = value_diag();
        }
        collect_regret(t_);
        fit_regret();
        if (eval_now) {
            double expl = exploitability(g_, tabular_average());
            if (cfg_.mode == Mode::Net)
                std::printf("  iter %4ld   expl(tabular-avg)=%.5f   V-rmse=%.4f (max %.3f)\n",
                            t_, expl, vd.first, vd.second);
            else
                std::printf("  iter %4ld   expl(tabular-avg)=%.5f\n", t_, expl);
            std::fflush(stdout);
        }
    }
}

}  // namespace escher
