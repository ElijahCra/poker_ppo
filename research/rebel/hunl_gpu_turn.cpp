#include "hunl_gpu_turn.h"
#include "hunl_value.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace rebel_hunl {

namespace {
constexpr double kTiny = 1e-12;
}

BatchTurnSolver::~BatchTurnSolver() = default;

BatchTurnSolver::BatchTurnSolver(TreeShape shape, std::vector<TurnSpec> specs,
                                 torch::Device device, torch::Dtype dtype,
                                 bool pcfr, bool pcfr_quad)
    : shape_(std::move(shape)), dev_(device), dt_(dtype),
      B_(static_cast<int>(specs.size())), pcfr_(pcfr),
      pcfr_quad_(pcfr_quad) {
    const auto& ct = ComboTable::get();
    const int n = kCombos;
    auto f = torch::TensorOptions().dtype(torch::kFloat);
    auto l = torch::TensorOptions().dtype(torch::kLong);

    // constants
    auto inc = torch::zeros({n, kCards}, f);
    auto ca = torch::empty({n}, l);
    auto cb = torch::empty({n}, l);
    auto mxc = torch::ones({kCards, n}, f);
    {
        auto ia = inc.accessor<float, 2>();
        auto pa = ca.accessor<int64_t, 1>();
        auto pb = cb.accessor<int64_t, 1>();
        auto mx = mxc.accessor<float, 2>();
        for (int i = 0; i < n; ++i) {
            ia[i][ct.cards[i][0]] = 1.0f;
            ia[i][ct.cards[i][1]] = 1.0f;
            pa[i] = ct.cards[i][0];
            pb[i] = ct.cards[i][1];
            mx[ct.cards[i][0]][i] = 0.0f;
            mx[ct.cards[i][1]][i] = 0.0f;
        }
    }

    const int M = static_cast<int>(shape_.nodes.size());
    bool any_allin = false;
    for (const auto& nd : shape_.nodes)
        if (nd.kind == HunlSolver::Node::AllinShowdown) any_allin = true;
    use_allin_op_ = any_allin && !std::getenv("REBEL_NO_ALLIN_OP");
    const bool build_runout = any_allin && !use_allin_op_;
    auto valid = torch::zeros({B_, n}, f);
    auto r0 = torch::zeros({B_, n}, f);
    auto r1 = torch::zeros({B_, n}, f);
    auto rok = torch::zeros({B_, kCards}, f);
    auto c0 = torch::zeros({B_, M}, f);
    auto c1 = torch::zeros({B_, M}, f);
    const long BR = static_cast<long>(B_) * kCards;
    auto perm = torch::zeros({BR, n}, l);
    auto prev1 = torch::zeros({BR, n}, l);
    auto segend = torch::zeros({BR, n}, l);
    auto poscard = torch::zeros({BR, kCards * 51}, l);
    auto ilowA = torch::zeros({BR, n}, l);
    auto ilowB = torch::zeros({BR, n}, l);
    auto iendA = torch::zeros({BR, n}, l);
    auto iendB = torch::zeros({BR, n}, l);
    auto itotA = torch::zeros({BR, n}, l);
    auto itotB = torch::zeros({BR, n}, l);

    for (int b = 0; b < B_; ++b) {
        const auto& sp = specs[static_cast<size_t>(b)];
        std::vector<uint8_t> vb;
        board_valid(sp.board.data(), 4, vb);
        double s0 = 0.0, s1 = 0.0;
        int nv4 = 0;
        for (int i = 0; i < n; ++i) {
            if (!vb[i]) continue;
            ++nv4;
            s0 += sp.r0[i];
            s1 += sp.r1[i];
        }
        auto va = valid.accessor<float, 2>();
        auto a0 = r0.accessor<float, 2>();
        auto a1 = r1.accessor<float, 2>();
        for (int i = 0; i < n; ++i) {
            va[b][i] = vb[i] ? 1.0f : 0.0f;
            if (vb[i]) {
                a0[b][i] = s0 > kTiny ? static_cast<float>(sp.r0[i] / s0)
                                      : static_cast<float>(1.0 / nv4);
                a1[b][i] = s1 > kTiny ? static_cast<float>(sp.r1[i] / s1)
                                      : static_cast<float>(1.0 / nv4);
            }
        }
        auto cc0 = c0.accessor<float, 2>();
        auto cc1 = c1.accessor<float, 2>();
        for (int m = 0; m < M; ++m) {
            cc0[b][m] = static_cast<float>(sp.node_contrib0[m]);
            cc1[b][m] = static_cast<float>(sp.node_contrib1[m]);
        }

        auto ro = rok.accessor<float, 2>();
        for (int c = 0; c < kCards; ++c) {
            bool on_board = false;
            for (int bd = 0; bd < 4; ++bd)
                if (sp.board[bd] == c) on_board = true;
            if (!on_board) ro[b][c] = 1.0f;
        }
        // fallback per-runout showdown structures (rows b*52+c) — only
        // when the allin operator is off. Board-card slots stay all-zero:
        // runout_ok_ kills their opp mass, so every gather lands on
        // zero-padded prefix cells and contributes nothing.
        for (int c = 0; build_runout && c < kCards; ++c) {
            bool on_board = false;
            for (int bd = 0; bd < 4; ++bd)
                if (sp.board[bd] == c) on_board = true;
            if (on_board) continue;
            const long row = static_cast<long>(b) * kCards + c;
            std::array<uint8_t, 5> b5{};
            for (int bd = 0; bd < 4; ++bd) b5[bd] = sp.board[bd];
            b5[4] = static_cast<uint8_t>(c);
            std::vector<uint8_t> v5;
            board_valid(b5.data(), 5, v5);
            std::vector<int> rank(n, -1), order;
            order.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (!v5[i]) continue;
                rank[i] = combo_rank(i, b5.data());
                order.push_back(i);
            }
            std::sort(order.begin(), order.end(),
                      [&](int x, int y) { return rank[x] < rank[y]; });
            const int nv = static_cast<int>(order.size());
            for (int i = 0; i < n; ++i)
                if (!v5[i]) order.push_back(i);
            auto pp = perm.accessor<int64_t, 2>();
            auto pe = prev1.accessor<int64_t, 2>();
            auto se = segend.accessor<int64_t, 2>();
            for (int pos = 0; pos < n; ++pos)
                pp[row][pos] = order[static_cast<size_t>(pos)];
            int gstart = 0, pos = 0;
            while (pos < nv) {
                int e = pos;
                while (e < nv && rank[order[static_cast<size_t>(e)]] ==
                                     rank[order[static_cast<size_t>(pos)]])
                    ++e;
                for (int k = pos; k < e; ++k) {
                    pe[row][k] = gstart;
                    se[row][k] = e;
                }
                gstart = e;
                pos = e;
            }
            for (int k = nv; k < n; ++k) {
                pe[row][k] = nv;
                se[row][k] = n;
            }
            std::array<std::vector<int>, kCards> posc;
            for (auto& v : posc) v.reserve(51);
            for (int p = 0; p < n; ++p) {
                const int cmb = order[static_cast<size_t>(p)];
                posc[ct.cards[cmb][0]].push_back(p);
                posc[ct.cards[cmb][1]].push_back(p);
            }
            auto pc = poscard.accessor<int64_t, 2>();
            for (int cc = 0; cc < kCards; ++cc)
                for (int j = 0; j < 51; ++j)
                    pc[row][cc * 51 + j] = posc[cc][static_cast<size_t>(j)];
            auto la = ilowA.accessor<int64_t, 2>();
            auto lb2 = ilowB.accessor<int64_t, 2>();
            auto ea = iendA.accessor<int64_t, 2>();
            auto eb = iendB.accessor<int64_t, 2>();
            auto ta = itotA.accessor<int64_t, 2>();
            auto tb = itotB.accessor<int64_t, 2>();
            auto count_below = [&](int cc, int64_t q) -> int64_t {
                const auto& v = posc[cc];
                return std::lower_bound(v.begin(), v.end(),
                                        static_cast<int>(q)) -
                       v.begin();
            };
            for (int p = 0; p < n; ++p) {
                const int cmb = order[static_cast<size_t>(p)];
                const int A = ct.cards[cmb][0], Bc = ct.cards[cmb][1];
                la[row][p] = A * 52 + count_below(A, pe[row][p]);
                lb2[row][p] = Bc * 52 + count_below(Bc, pe[row][p]);
                ea[row][p] = A * 52 + count_below(A, se[row][p]);
                eb[row][p] = Bc * 52 + count_below(Bc, se[row][p]);
                ta[row][p] = A * 52 + 51;
                tb[row][p] = Bc * 52 + 51;
            }
        }
    }

    inc_ = inc.to(dev_).to(dt_);
    card_a_ = ca.to(dev_);
    card_b_ = cb.to(dev_);
    mask_xc_ = mxc.to(dev_).to(dt_);
    valid_ = valid.to(dev_).to(dt_);
    r0_ = (r0 * valid).to(dev_).to(dt_);
    r1_ = (r1 * valid).to(dev_).to(dt_);
    r0c_ = (r0 * valid).contiguous();
    r1c_ = (r1 * valid).contiguous();
    runout_ok_ = rok.to(dev_).to(dt_).unsqueeze(2);        // [B, 52, 1]
    c0_ = c0.to(dev_).to(dt_);
    c1_ = c1.to(dev_).to(dt_);
    if (build_runout) {
        perm52_ = perm.to(dev_);
        prev52_ = prev1.to(dev_);
        seg52_ = segend.to(dev_);
        poscard52_ = poscard.to(dev_);
        ilowA52_ = ilowA.to(dev_);
        ilowB52_ = ilowB.to(dev_);
        iendA52_ = iendA.to(dev_);
        iendB52_ = iendB.to(dev_);
        itotA52_ = itotA.to(dev_);
        itotB52_ = itotB.to(dev_);
    }
    if (use_allin_op_) {
        // Op[x][y] = half_pot/44 · Σ_c sign(x beats y on runout c), exact
        // card removal (x∋c / y∋c rows excluded; overlapping pairs 0) —
        // higher combo_rank wins, ties contribute 0 (2·win + tie − tot).
        int allin_node = -1;
        for (int m = 0; m < M; ++m)
            if (shape_.nodes[static_cast<size_t>(m)].kind ==
                HunlSolver::Node::AllinShowdown)
                allin_node = m;
        // accumulated in DOUBLE: an f32-quantized operator inside an f64
        // solve perturbs the trajectory enough to flip RM+ clamps and
        // land on a different equilibrium (phase-B check failure)
        auto opT = torch::zeros(
            {B_, static_cast<long>(n), static_cast<long>(n)},
            torch::kDouble);
        std::atomic<int> nb{0};
        auto build = [&]() {
            std::vector<int> rank(static_cast<size_t>(n));
            std::vector<int> live;
            live.reserve(static_cast<size_t>(n));
            while (true) {
                const int b = nb.fetch_add(1);
                if (b >= B_) break;
                const auto& sp = specs[static_cast<size_t>(b)];
                double* op = opT.data_ptr<double>() +
                             static_cast<size_t>(b) * n * n;
                for (int c = 0; c < kCards; ++c) {
                    bool on_board = false;
                    for (int bd = 0; bd < 4; ++bd)
                        if (sp.board[static_cast<size_t>(bd)] == c)
                            on_board = true;
                    if (on_board) continue;
                    std::array<uint8_t, 5> b5{};
                    for (int bd = 0; bd < 4; ++bd)
                        b5[static_cast<size_t>(bd)] =
                            sp.board[static_cast<size_t>(bd)];
                    b5[4] = static_cast<uint8_t>(c);
                    std::vector<uint8_t> v5;
                    board_valid(b5.data(), 5, v5);
                    live.clear();
                    for (int i = 0; i < n; ++i)
                        if (v5[i]) {
                            rank[static_cast<size_t>(i)] =
                                combo_rank(i, b5.data());
                            live.push_back(i);
                        }
                    for (int xi : live) {
                        const int xa = ct.cards[xi][0], xb = ct.cards[xi][1];
                        const int rx = rank[static_cast<size_t>(xi)];
                        double* row = op + static_cast<size_t>(xi) * n;
                        for (int yi : live) {
                            const int ya = ct.cards[yi][0],
                                      yb = ct.cards[yi][1];
                            if (ya == xa || ya == xb || yb == xa ||
                                yb == xb)
                                continue;
                            const int ry = rank[static_cast<size_t>(yi)];
                            if (rx > ry)
                                row[yi] += 1.0;
                            else if (rx < ry)
                                row[yi] -= 1.0;
                        }
                    }
                }
                const double scale =
                    static_cast<double>(sp.node_contrib0[static_cast<size_t>(
                        allin_node)]) / 44.0;
                for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i)
                    op[i] *= scale;
            }
        };
        int W = static_cast<int>(std::thread::hardware_concurrency());
        W = std::max(1, std::min(W, B_));
        if (W == 1) {
            build();
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(W));
            for (int w = 0; w < W; ++w) pool.emplace_back(build);
            for (auto& th : pool) th.join();
        }
        allin_op_ = opT.to(dev_).to(dt_);
    }

    regret_.resize(shape_.nodes.size());
    cum_.resize(shape_.nodes.size());
    pred_.resize(shape_.nodes.size());
    for (size_t m = 0; m < shape_.nodes.size(); ++m) {
        const auto& nd = shape_.nodes[m];
        if (nd.kind == HunlSolver::Node::Decision) {
            const long A = static_cast<long>(nd.acts.size());
            regret_[m] = torch::zeros(
                {B_, A, n}, torch::TensorOptions().dtype(dt_).device(dev_));
            cum_[m] = torch::zeros_like(regret_[m]);
            if (pcfr_) pred_[m] = torch::zeros_like(regret_[m]);
        } else if (nd.kind == HunlSolver::Node::StreetEnd) {
            leaf_nodes_.push_back(static_cast<int>(m));
            leaf_v_.push_back(torch::zeros(
                {B_, kCards, 2, n},
                torch::TensorOptions().dtype(dt_).device(dev_)));
        }
    }
    // env-action paths root → leaf (for callers that need to position an
    // env at the leaf; mirrors HunlSolver::Node::path)
    leaf_path_.resize(leaf_nodes_.size());
    for (size_t j = 0; j < leaf_nodes_.size(); ++j) {
        // walk up via a parent scan (trees are tiny)
        std::vector<int> rev;
        int cur = leaf_nodes_[j];
        while (cur != 0) {
            for (int m = 0; m < M; ++m) {
                const auto& nd = shape_.nodes[static_cast<size_t>(m)];
                bool hit = false;
                for (size_t k = 0; k < nd.child.size(); ++k)
                    if (nd.child[k] == cur) {
                        rev.push_back(nd.acts[k]);
                        cur = m;
                        hit = true;
                        break;
                    }
                if (hit) break;
            }
        }
        leaf_path_[j].assign(rev.rbegin(), rev.rend());
    }
    w_dev_ = torch::ones({}, torch::TensorOptions().dtype(dt_).device(dev_));
    root_acc_[0] = torch::zeros(
        {B_, n}, torch::TensorOptions().dtype(dt_).device(dev_));
    root_acc_[1] = torch::zeros_like(root_acc_[0]);
}

torch::Tensor BatchTurnSolver::policies(int node, bool average) {
    const auto& nd = shape_.nodes[static_cast<size_t>(node)];
    torch::Tensor src;
    if (average)
        src = cum_[static_cast<size_t>(node)];
    else if (pcfr_)
        // PCFR+: current policy regret-matches over [R + m]^+
        src = regret_[static_cast<size_t>(node)] +
              pred_[static_cast<size_t>(node)];
    else
        src = regret_[static_cast<size_t>(node)];
    auto pos = torch::clamp_min(src, 0.0f);
    auto s = pos.sum(1, /*keepdim=*/true);
    const float uni = 1.0f / static_cast<float>(nd.acts.size());
    return torch::where(s > 1e-12f, pos / s.clamp_min(1e-12f),
                        torch::full_like(pos, uni));
}

torch::Tensor BatchTurnSolver::fold_cfv_t(int node, int upd,
                                          const torch::Tensor& opp) {
    auto S = opp.sum(1, /*keepdim=*/true);
    auto Sc = torch::matmul(opp, inc_);
    auto mass = S - Sc.index_select(1, card_a_) -
                Sc.index_select(1, card_b_) + opp;
    const int folder = shape_.nodes[static_cast<size_t>(node)].player;
    auto u = (upd == folder)
        ? -(upd == 0 ? c0_ : c1_).select(1, node)
        : (upd == 0 ? c1_ : c0_).select(1, node);
    return mass * u.unsqueeze(1) * valid_;
}

torch::Tensor BatchTurnSolver::allin_cfv_t(int node,
                                           const torch::Tensor& opp) {
    if (use_allin_op_) {
        (void)node;   // all allin nodes share full-stack contribs
        return torch::bmm(allin_op_, opp.unsqueeze(2)).squeeze(2) * valid_;
    }
    const int n = kCombos;
    // runouts ride the batch dim: opp masked per runout card → [B·52, n]
    auto opp52 = (opp.unsqueeze(1) * mask_xc_ * runout_ok_)
                     .reshape({static_cast<long>(B_) * kCards, n});
    auto ws = opp52.gather(1, perm52_);
    auto cs = ws.cumsum(1);
    auto cs0 = torch::cat(
        {torch::zeros({ws.size(0), 1}, ws.options()), cs}, 1);
    auto wsc = ws.gather(1, poscard52_)
                   .reshape({ws.size(0), kCards, 51});
    auto csc = wsc.cumsum(2);
    auto csc0 = torch::cat(
        {torch::zeros({ws.size(0), kCards, 1}, ws.options()), csc}, 2)
                    .reshape({ws.size(0), kCards * 52});

    auto lowT = cs0.gather(1, prev52_);
    auto lowA = csc0.gather(1, ilowA52_);
    auto lowB = csc0.gather(1, ilowB52_);
    auto win = lowT - lowA - lowB;
    auto endT = cs0.gather(1, seg52_);
    auto endA = csc0.gather(1, iendA52_);
    auto endB = csc0.gather(1, iendB52_);
    auto tie = (endT - lowT) - (endA - lowA) - (endB - lowB) + ws;
    auto S = cs.select(1, n - 1).unsqueeze(1);
    auto ScA = csc0.gather(1, itotA52_);
    auto ScB = csc0.gather(1, itotB52_);
    auto tot = S - ScA - ScB + ws;

    auto half_pot = c0_.select(1, node)
                        .unsqueeze(1)
                        .expand({B_, static_cast<long>(kCards)})
                        .reshape({static_cast<long>(B_) * kCards, 1});
    auto cfv_sorted = half_pot * (2.0f * win + tie - tot);
    auto cfv52 = torch::zeros_like(cfv_sorted);
    cfv52.scatter_(1, perm52_, cfv_sorted);
    // mask x∋c, kill board-card slots, average over the 44 per-pair runouts
    auto out = (cfv52.reshape({B_, kCards, n}) * mask_xc_ * runout_ok_)
                   .sum(1) /
               44.0;
    return out * valid_;
}

torch::Tensor BatchTurnSolver::street_end_cfv_t(int node, int upd,
                                                const torch::Tensor& opp) {
    int j = -1;
    for (size_t k = 0; k < leaf_nodes_.size(); ++k)
        if (leaf_nodes_[k] == node) j = static_cast<int>(k);
    TORCH_CHECK(j >= 0, "street_end_cfv_t: not a leaf node");
    auto opp52 = opp.unsqueeze(1) * mask_xc_ * runout_ok_;   // [B, 52, n]
    auto S = opp52.sum(2, /*keepdim=*/true);                 // [B, 52, 1]
    auto Sc = torch::matmul(opp52, inc_);                    // [B, 52, 52]
    auto mass = S - Sc.index_select(2, card_a_) -
                Sc.index_select(2, card_b_) + opp52;
    auto v = leaf_v_[static_cast<size_t>(j)].select(2, upd); // [B, 52, n]
    return (v * mass * mask_xc_).sum(1) / 44.0 * valid_;
}

torch::Tensor BatchTurnSolver::walk(int node, int upd, bool update,
                                    torch::Tensor my_reach,
                                    torch::Tensor opp_reach) {
    const auto& nd = shape_.nodes[static_cast<size_t>(node)];
    switch (static_cast<HunlSolver::Node::Kind>(nd.kind)) {
    case HunlSolver::Node::Fold:
        return fold_cfv_t(node, upd, opp_reach);
    case HunlSolver::Node::Showdown:
        TORCH_CHECK(false, "turn tree has no single-street Showdown");
    case HunlSolver::Node::AllinShowdown:
        return allin_cfv_t(node, opp_reach);
    case HunlSolver::Node::StreetEnd:
        return street_end_cfv_t(node, upd, opp_reach);
    case HunlSolver::Node::Decision:
        break;
    }
    const int A = static_cast<int>(nd.acts.size());
    auto sig = policies(node, /*average=*/!update);
    if (nd.player == upd) {
        std::vector<torch::Tensor> cfv_a(A);
        for (int k = 0; k < A; ++k)
            cfv_a[k] = walk(nd.child[k], upd, update,
                            my_reach * sig.select(1, k), opp_reach);
        auto stacked = torch::stack(cfv_a, 1);
        auto v = (sig * stacked).sum(1);
        if (update) {
            if (pcfr_) {
                // inst kept as a tensor: regret and pred both need it,
                // with the CPU path's (r + inst) association
                auto inst = stacked - v.unsqueeze(1);
                regret_[static_cast<size_t>(node)].add_(inst).clamp_min_(
                    0.0);
                pred_[static_cast<size_t>(node)].copy_(inst);
            } else {
                // river solver's exact in-place order (equivalence
                // discipline: RM⁺'s clamp is order-sensitive)
                regret_[static_cast<size_t>(node)]
                    .add_(stacked)
                    .sub_(v.unsqueeze(1))
                    .clamp_min_(0.0);
            }
            cum_[static_cast<size_t>(node)].add_(
                (w_dev_ * my_reach.unsqueeze(1)) * sig);
        }
        return v;
    }
    torch::Tensor cfv;
    for (int k = 0; k < A; ++k) {
        auto cv = walk(nd.child[k], upd, update, my_reach,
                       opp_reach * sig.select(1, k));
        cfv = cfv.defined() ? cfv + cv : cv;
    }
    return cfv;
}

void BatchTurnSolver::iterate_body() {
    for (int upd = 0; upd < 2; ++upd) {
        auto mine = upd == 0 ? r0_ : r1_;
        auto opp = upd == 0 ? r1_ : r0_;
        auto cfv = walk(0, upd, /*update=*/true, mine, opp);
        root_acc_[static_cast<size_t>(upd)].add_(cfv);
    }
}

void BatchTurnSolver::iterate(int t) {
    torch::NoGradGuard ng;
    w_dev_.fill_(weight(t));
    iterate_body();
    ++rv_n_;
}

void BatchTurnSolver::solve(int T, int refresh_every,
                            const std::function<void(int)>& refresh) {
    torch::NoGradGuard ng;
    auto sched = [&](int t) {
        return t == 1 || refresh_every <= 1 || t % refresh_every == 0;
    };
    const bool no_graph = std::getenv("REBEL_NO_CUDA_GRAPH") != nullptr;
    if (!dev_.is_cuda() || T < 8 || no_graph) {
        for (int t = 1; t <= T; ++t) {
            if (sched(t)) refresh(t);
            iterate(t);
        }
        return;
    }
    // warmup (allocator settles), then capture ONE full iteration; per
    // remaining iteration: one weight fill_ + one replay. Refreshes stay
    // OUTSIDE the graph: set_leaf_values copies into the same storage the
    // captured walk reads, so replays see fresh leaf values.
    try {
        auto stream = at::cuda::getStreamFromPool();
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            for (int t = 1; t <= 3 && t <= T; ++t) {
                if (sched(t)) refresh(t);
                iterate(t);
            }
            at::cuda::getCurrentCUDAStream().synchronize();
            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            iterate_body();
            graph_->capture_end();
            at::cuda::getCurrentCUDAStream().synchronize();
        }
        // capture RECORDS without executing: iteration 4 onward replays
        for (int t = 4; t <= T; ++t) {
            if (sched(t)) refresh(t);
            w_dev_.fill_(weight(t));
            graph_->replay();
            ++rv_n_;
        }
        torch::cuda::synchronize();
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[gpu-turn] CUDA graph capture failed (%s) — eager "
                     "fallback\n", e.what());
        graph_.reset();
        for (int t = 4; t <= T; ++t) {
            if (sched(t)) refresh(t);
            iterate(t);
        }
    }
}

void refresh_turn_leaves(
    BatchTurnSolver& s, const std::vector<TurnSpec>& specs,
    const std::vector<HunlNetOracle*>& oracles, int threads,
    TurnRefreshWorkspace* ws) {
    const int B = s.batch();
    const auto& ct = ComboTable::get();
    const int L = static_cast<int>(s.leaf_nodes().size());
    const int C = kCards - 4;   // off-board river cards per spec
    // all leaf beliefs in ONE policy download + CPU walks (per-leaf
    // device walks measured dominant: launch overhead + syncs)
    std::vector<std::vector<std::array<std::vector<double>, 2>>> reach;
    s.all_leaf_reaches(reach, threads);
    const long R = static_cast<long>(B) * L * C;
    // host buffers from the workspace when given — re-allocating ~450MB
    // per refresh (feature block + value tensors) costs more than the
    // math. Leaf-value dtype follows the solver (net outputs are f32-
    // quantized either way; f64 kept for the equivalence check's bars).
    TurnRefreshWorkspace local;
    TurnRefreshWorkspace& w = ws ? *ws : local;
    const auto vdt =
        s.dtype() == torch::kDouble ? torch::kDouble : torch::kFloat;
    if (!w.X.defined() || w.X.size(0) != R)
        w.X = torch::empty(
            {R, HunlFeaturizer::kDim},
            torch::TensorOptions().dtype(torch::kFloat).pinned_memory(
                torch::cuda::is_available()));
    if (static_cast<int>(w.vals.size()) != L ||
        (L > 0 && (w.vals[0].size(0) != B ||
                   w.vals[0].scalar_type() != vdt))) {
        w.vals.assign(static_cast<size_t>(L), {});
        for (int j = 0; j < L; ++j)
            w.vals[static_cast<size_t>(j)] = torch::zeros(
                {B, static_cast<long>(kCards), 2,
                 static_cast<long>(kCombos)},
                torch::TensorOptions().dtype(vdt));
    }
    auto& vals = w.vals;
    float* xp = w.X.data_ptr<float>();
    std::atomic<int> next{0};
    auto feat_work = [&]() {
        HunlPBS lb;
        while (true) {
            const int b = next.fetch_add(1);
            if (b >= B) break;
            const auto& sp = specs[static_cast<size_t>(b)];
            for (int j = 0; j < L; ++j) {
                const int leaf = s.leaf_nodes()[static_cast<size_t>(j)];
                const double pot = static_cast<double>(
                    sp.node_contrib0[static_cast<size_t>(leaf)] +
                    sp.node_contrib1[static_cast<size_t>(leaf)]);
                int q = 0;
                for (int c = 0; c < kCards; ++c) {
                    bool on = false;
                    for (int t = 0; t < 4; ++t)
                        if (sp.board[static_cast<size_t>(t)] == c)
                            on = true;
                    if (on) continue;
                    lb.r0 = reach[static_cast<size_t>(j)]
                                 [static_cast<size_t>(b)][0];
                    lb.r1 = reach[static_cast<size_t>(j)]
                                 [static_cast<size_t>(b)][1];
                    for (int x = 0; x < kCombos; ++x)
                        if (ct.cards[x][0] == c || ct.cards[x][1] == c)
                            lb.r0[static_cast<size_t>(x)] =
                                lb.r1[static_cast<size_t>(x)] = 0.0;
                    const long row =
                        (static_cast<long>(b) * L + j) * C + q;
                    oracles[static_cast<size_t>(b)]->river_row_features(
                        sp.board.data(), static_cast<uint8_t>(c), pot, lb,
                        xp + static_cast<size_t>(row) *
                                 HunlFeaturizer::kDim);
                    ++q;
                }
            }
        }
    };
    int W = threads > 0
        ? threads
        : static_cast<int>(std::thread::hardware_concurrency());
    W = std::max(1, std::min(W, B));
    auto fan_out = [&](const std::function<void()>& fn) {
        next.store(0);
        if (W == 1) {
            fn();
            return;
        }
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(W));
        for (int w = 0; w < W; ++w) pool.emplace_back(fn);
        for (auto& th : pool) th.join();
    };
    using clk = std::chrono::steady_clock;
    const bool prof = std::getenv("REBEL_TURN_PROFILE") != nullptr;
    auto p0 = clk::now();
    fan_out(feat_work);
    auto p1 = clk::now();
    // one CHUNKED forward for the whole refresh (pot units): bounded
    // device transients — a single all-rows forward at large B collides
    // with the solver state + graph pool and stalls on allocator
    // eviction (measured 0.4s → 11s per refresh at B=128 on 8GB)
    if (!w.Y.defined() || w.Y.size(0) != R)
        w.Y = torch::empty({R, 2L * kCombos}, torch::kFloat);
    const long chunk = 8192;
    for (long off = 0; off < R; off += chunk) {
        const long len = std::min(chunk, R - off);
        w.Y.narrow(0, off, len)
            .copy_(oracles[0]->values_forward(w.X.narrow(0, off, len)));
    }
    const float* yp = w.Y.data_ptr<float>();
    auto p2 = clk::now();
    // pass 2 (threads over specs): scatter into the per-leaf card slots,
    // scaled to chips
    auto scatter_work = [&]() {
        while (true) {
            const int b = next.fetch_add(1);
            if (b >= B) break;
            const auto& sp = specs[static_cast<size_t>(b)];
            for (int j = 0; j < L; ++j) {
                const int leaf = s.leaf_nodes()[static_cast<size_t>(j)];
                const double pot = static_cast<double>(
                    sp.node_contrib0[static_cast<size_t>(leaf)] +
                    sp.node_contrib1[static_cast<size_t>(leaf)]);
                auto& vt = vals[static_cast<size_t>(j)];
                int q = 0;
                for (int c = 0; c < kCards; ++c) {
                    bool on = false;
                    for (int t = 0; t < 4; ++t)
                        if (sp.board[static_cast<size_t>(t)] == c)
                            on = true;
                    if (on) continue;
                    const float* row =
                        yp + ((static_cast<long>(b) * L + j) * C + q) * 2 *
                                 kCombos;
                    const size_t base =
                        ((static_cast<size_t>(b) * kCards + c) * 2) *
                        kCombos;
                    if (vdt == torch::kDouble) {
                        double* dst = vt.data_ptr<double>() + base;
                        for (int i = 0; i < 2 * kCombos; ++i)
                            dst[i] = static_cast<double>(row[i]) * pot;
                    } else {
                        float* dst = vt.data_ptr<float>() + base;
                        for (int i = 0; i < 2 * kCombos; ++i)
                            dst[i] = static_cast<float>(
                                static_cast<double>(row[i]) * pot);
                    }
                    ++q;
                }
            }
        }
    };
    fan_out(scatter_work);
    for (int j = 0; j < L; ++j)
        s.set_leaf_values(j, vals[static_cast<size_t>(j)]);
    if (prof) {
        auto p3 = clk::now();
        auto secs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        std::printf("    [refresh] feat %.3fs fwd %.3fs scatter+up %.3fs "
                    "(W=%d R=%ld)\n", secs(p0, p1), secs(p1, p2),
                    secs(p2, p3), W, R);
    }
}

void BatchTurnSolver::leaf_reaches(
    int leaf_idx, std::vector<std::array<std::vector<double>, 2>>& out) {
    torch::NoGradGuard ng;
    auto r0 = r0_.clone();
    auto r1 = r1_.clone();
    int node = 0;
    for (int a : leaf_path_[static_cast<size_t>(leaf_idx)]) {
        const auto& nd = shape_.nodes[static_cast<size_t>(node)];
        int k = -1;
        for (size_t q = 0; q < nd.acts.size(); ++q)
            if (nd.acts[q] == a) k = static_cast<int>(q);
        TORCH_CHECK(k >= 0, "leaf_reaches: path action not in tree");
        auto sig = policies(node, /*average=*/true).select(1, k);
        (nd.player == 0 ? r0 : r1).mul_(sig);
        node = nd.child[static_cast<size_t>(k)];
    }
    auto h0 = r0.to(torch::kCPU).to(torch::kDouble).contiguous();
    auto h1 = r1.to(torch::kCPU).to(torch::kDouble).contiguous();
    out.assign(static_cast<size_t>(B_), {});
    auto a0 = h0.accessor<double, 2>();
    auto a1 = h1.accessor<double, 2>();
    for (int b = 0; b < B_; ++b) {
        out[static_cast<size_t>(b)][0].assign(kCombos, 0.0);
        out[static_cast<size_t>(b)][1].assign(kCombos, 0.0);
        for (int i = 0; i < kCombos; ++i) {
            out[static_cast<size_t>(b)][0][static_cast<size_t>(i)] =
                a0[b][i];
            out[static_cast<size_t>(b)][1][static_cast<size_t>(i)] =
                a1[b][i];
        }
    }
}

void BatchTurnSolver::all_leaf_reaches(
    std::vector<std::vector<std::array<std::vector<double>, 2>>>& out,
    int threads) {
    torch::NoGradGuard ng;
    const int L = static_cast<int>(leaf_nodes_.size());
    out.assign(static_cast<size_t>(L), {});
    // one bulk download of the average strategies (cum_) for the decision
    // nodes on any leaf path
    std::vector<torch::Tensor> pol(shape_.nodes.size());
    std::vector<char> need(shape_.nodes.size(), 0);
    for (int j = 0; j < L; ++j) {
        int node = 0;
        for (int a : leaf_path_[static_cast<size_t>(j)]) {
            need[static_cast<size_t>(node)] = 1;
            const auto& nd = shape_.nodes[static_cast<size_t>(node)];
            for (size_t q = 0; q < nd.acts.size(); ++q)
                if (nd.acts[q] == a) {
                    node = nd.child[q];
                    break;
                }
        }
    }
    for (size_t m = 0; m < shape_.nodes.size(); ++m)
        if (need[m])
            pol[m] =
                cum_[m].to(torch::kCPU).to(torch::kDouble).contiguous();
    const float* pr0 = r0c_.data_ptr<float>();
    const float* pr1 = r1c_.data_ptr<float>();
    for (int j = 0; j < L; ++j) {
        out[static_cast<size_t>(j)].assign(static_cast<size_t>(B_), {});
        for (int b = 0; b < B_; ++b)
            for (int p = 0; p < 2; ++p)
                out[static_cast<size_t>(j)][static_cast<size_t>(b)]
                   [static_cast<size_t>(p)].assign(kCombos, 0.0);
    }
    std::atomic<int> nextb{0};
    auto work = [&]() {
        while (true) {
            const int b = nextb.fetch_add(1);
            if (b >= B_) break;
            for (int j = 0; j < L; ++j) {
                auto& r0v = out[static_cast<size_t>(j)]
                               [static_cast<size_t>(b)][0];
                auto& r1v = out[static_cast<size_t>(j)]
                               [static_cast<size_t>(b)][1];
                for (int i = 0; i < kCombos; ++i) {
                    r0v[static_cast<size_t>(i)] = static_cast<double>(
                        pr0[static_cast<size_t>(b) * kCombos + i]);
                    r1v[static_cast<size_t>(i)] = static_cast<double>(
                        pr1[static_cast<size_t>(b) * kCombos + i]);
                }
                int node = 0;
                for (int a : leaf_path_[static_cast<size_t>(j)]) {
                    const auto& nd =
                        shape_.nodes[static_cast<size_t>(node)];
                    const int A = static_cast<int>(nd.acts.size());
                    int k = -1;
                    for (size_t q = 0; q < nd.acts.size(); ++q)
                        if (nd.acts[q] == a) k = static_cast<int>(q);
                    const double* pm =
                        pol[static_cast<size_t>(node)].data_ptr<double>() +
                        static_cast<size_t>(b) * A * kCombos;
                    auto& mine = nd.player == 0 ? r0v : r1v;
                    for (int x = 0; x < kCombos; ++x) {
                        if (mine[static_cast<size_t>(x)] <= 0.0) continue;
                        double s = 0.0;
                        for (int q = 0; q < A; ++q) {
                            const double v = pm[static_cast<size_t>(q) *
                                                    kCombos + x];
                            s += v > 0.0 ? v : 0.0;
                        }
                        const double pk =
                            pm[static_cast<size_t>(k) * kCombos + x];
                        mine[static_cast<size_t>(x)] *=
                            s > 1e-12 ? std::max(0.0, pk) / s : 1.0 / A;
                    }
                    node = nd.child[static_cast<size_t>(k)];
                }
            }
        }
    };
    int W = threads > 0
        ? threads
        : static_cast<int>(std::thread::hardware_concurrency());
    W = std::max(1, std::min(W, B_));
    if (W == 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(W));
        for (int w = 0; w < W; ++w) pool.emplace_back(work);
        for (auto& th : pool) th.join();
    }
}

void BatchTurnSolver::set_leaf_values(int leaf_idx,
                                      const torch::Tensor& vals) {
    TORCH_CHECK(vals.dim() == 4 && vals.size(0) == B_ &&
                    vals.size(1) == kCards && vals.size(2) == 2 &&
                    vals.size(3) == kCombos,
                "set_leaf_values: want [B, 52, 2, n]");
    leaf_v_[static_cast<size_t>(leaf_idx)].copy_(
        vals.to(dev_).to(dt_));
}

bool BatchTurnSolver::avg_root_values(
    std::vector<std::array<std::vector<double>, 2>>& v,
    std::vector<std::array<std::vector<double>, 2>>& mask) {
    if (rv_n_ <= 0) return false;
    v.assign(static_cast<size_t>(B_), {});
    mask.assign(static_cast<size_t>(B_), {});
    auto val = valid_.to(torch::kCPU).to(torch::kDouble);
    auto va = val.accessor<double, 2>();
    for (int p = 0; p < 2; ++p) {
        auto acc = root_acc_[static_cast<size_t>(p)]
                       .to(torch::kCPU)
                       .to(torch::kDouble);
        auto own = (p == 0 ? r0_ : r1_).to(torch::kCPU).to(torch::kDouble);
        auto oppt = (p == 0 ? r1_ : r0_).to(torch::kCPU).to(torch::kDouble);
        auto aa = acc.accessor<double, 2>();
        auto ao = own.accessor<double, 2>();
        auto ap = oppt.accessor<double, 2>();
        for (int b = 0; b < B_; ++b) {
            v[static_cast<size_t>(b)][static_cast<size_t>(p)].assign(
                kCombos, 0.0);
            mask[static_cast<size_t>(b)][static_cast<size_t>(p)].assign(
                kCombos, 0.0);
            // compat_mass on CPU doubles, same as the CPU solver
            std::vector<double> opp(kCombos), vd(kCombos);
            for (int i = 0; i < kCombos; ++i) {
                opp[static_cast<size_t>(i)] = ap[b][i];
                vd[static_cast<size_t>(i)] = va[b][i];
            }
            const auto& ct = ComboTable::get();
            double S = 0.0, Sc[kCards] = {};
            for (int i = 0; i < kCombos; ++i) {
                if (vd[static_cast<size_t>(i)] < 0.5) continue;
                S += opp[static_cast<size_t>(i)];
                Sc[ct.cards[i][0]] += opp[static_cast<size_t>(i)];
                Sc[ct.cards[i][1]] += opp[static_cast<size_t>(i)];
            }
            for (int x = 0; x < kCombos; ++x) {
                if (vd[static_cast<size_t>(x)] < 0.5) continue;
                const double m = S - Sc[ct.cards[x][0]] -
                                 Sc[ct.cards[x][1]] +
                                 opp[static_cast<size_t>(x)];
                if (ao[b][x] <= kTiny || m <= 1e-6) continue;
                v[static_cast<size_t>(b)][static_cast<size_t>(p)]
                 [static_cast<size_t>(x)] =
                     aa[b][x] / static_cast<double>(rv_n_) / m;
                mask[static_cast<size_t>(b)][static_cast<size_t>(p)]
                    [static_cast<size_t>(x)] = 1.0;
            }
        }
    }
    return true;
}

}  // namespace rebel_hunl
