#include "hunl_gpu.h"
#include "hunl_fused.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace rebel_hunl {

namespace {
constexpr double kTiny = 1e-12;
}

BatchRiverSolver::~BatchRiverSolver() = default;

TreeShape TreeShape::from(const HunlSolver& s) {
    TreeShape sh;
    sh.nodes.reserve(s.nodes().size());
    for (const auto& nd : s.nodes()) {
        TreeShape::Node n;
        n.kind = static_cast<int>(nd.kind);
        n.player = nd.player;
        n.acts = nd.acts;
        n.child = nd.child;
        sh.nodes.push_back(std::move(n));
        sh.signature += std::to_string(n.kind) + ":" +
                        std::to_string(n.player) + ":" +
                        std::to_string(n.acts.size()) + ";";
    }
    return sh;
}

BatchRiverSolver::BatchRiverSolver(TreeShape shape,
                                   std::vector<RiverSpec> specs,
                                   torch::Device device, torch::Dtype dtype)
    : shape_(std::move(shape)), dev_(device), dt_(dtype),
      B_(static_cast<int>(specs.size())) {
    const auto& ct = ComboTable::get();
    const int n = kCombos;
    auto f = torch::TensorOptions().dtype(torch::kFloat);
    auto l = torch::TensorOptions().dtype(torch::kLong);

    // constants
    auto inc = torch::zeros({n, kCards}, f);
    auto ca = torch::empty({n}, l);
    auto cb = torch::empty({n}, l);
    {
        auto ia = inc.accessor<float, 2>();
        auto pa = ca.accessor<int64_t, 1>();
        auto pb = cb.accessor<int64_t, 1>();
        for (int i = 0; i < n; ++i) {
            ia[i][ct.cards[i][0]] = 1.0f;
            ia[i][ct.cards[i][1]] = 1.0f;
            pa[i] = ct.cards[i][0];
            pb[i] = ct.cards[i][1];
        }
    }

    // per-spec: validity, normalized ranges, showdown sort/segments, contribs
    auto valid = torch::zeros({B_, n}, f);
    auto r0 = torch::zeros({B_, n}, f);
    auto r1 = torch::zeros({B_, n}, f);
    auto perm = torch::zeros({B_, n}, l);
    auto prev1 = torch::zeros({B_, n}, l);
    auto segend = torch::zeros({B_, n}, l);
    auto poscard = torch::zeros({B_, kCards * 51}, l);
    auto ilowA = torch::zeros({B_, n}, l);
    auto ilowB = torch::zeros({B_, n}, l);
    auto iendA = torch::zeros({B_, n}, l);
    auto iendB = torch::zeros({B_, n}, l);
    auto itotA = torch::zeros({B_, n}, l);
    auto itotB = torch::zeros({B_, n}, l);
    const int M = static_cast<int>(shape_.nodes.size());
    auto c0 = torch::zeros({B_, M}, f);
    auto c1 = torch::zeros({B_, M}, f);

    for (int b = 0; b < B_; ++b) {
        const auto& sp = specs[static_cast<size_t>(b)];
        std::vector<uint8_t> vb;
        board_valid(sp.board.data(), 5, vb);
        // ranks + sorted order (valid first, ascending; invalid padded last)
        std::vector<int> rank(n, -1), order;
        order.reserve(n);
        double s0 = 0.0, s1 = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!vb[i]) continue;
            rank[i] = combo_rank(i, sp.board.data());
            order.push_back(i);
            s0 += sp.r0[i];
            s1 += sp.r1[i];
        }
        std::sort(order.begin(), order.end(),
                  [&](int x, int y) { return rank[x] < rank[y]; });
        const int nv = static_cast<int>(order.size());
        for (int i = 0; i < n; ++i)
            if (!vb[i]) order.push_back(i);   // pad (zero mass throughout)

        auto va = valid.accessor<float, 2>();
        auto a0 = r0.accessor<float, 2>();
        auto a1 = r1.accessor<float, 2>();
        auto pp = perm.accessor<int64_t, 2>();
        auto pe = prev1.accessor<int64_t, 2>();
        auto se = segend.accessor<int64_t, 2>();
        for (int i = 0; i < n; ++i) {
            va[b][i] = vb[i] ? 1.0f : 0.0f;
            if (vb[i]) {
                a0[b][i] = s0 > kTiny ? static_cast<float>(sp.r0[i] / s0)
                                      : static_cast<float>(1.0 / nv);
                a1[b][i] = s1 > kTiny ? static_cast<float>(sp.r1[i] / s1)
                                      : static_cast<float>(1.0 / nv);
            }
        }
        for (int pos = 0; pos < n; ++pos)
            pp[b][pos] = order[static_cast<size_t>(pos)];
        // segment boundaries over the sorted VALID prefix; pads form one
        // trailing zero-mass segment.
        int gstart = 0;
        int pos = 0;
        while (pos < nv) {
            int e = pos;
            while (e < nv &&
                   rank[order[static_cast<size_t>(e)]] ==
                       rank[order[static_cast<size_t>(pos)]])
                ++e;
            for (int k = pos; k < e; ++k) {
                pe[b][k] = gstart;   // == index into zero-padded cumsum
                se[b][k] = e;
            }
            gstart = e;
            pos = e;
        }
        for (int k = nv; k < n; ++k) {   // pad segment
            pe[b][k] = nv;
            se[b][k] = n;
        }
        // sparse per-card structure: ascending positions per card (each
        // card is in exactly 51 combos, pads included with zero mass) and
        // constant boundary counts per row (boundaries are per-spec
        // constants, so lower_bound resolves them here on CPU once)
        {
            std::array<std::vector<int>, kCards> posc;
            for (auto& v : posc) v.reserve(51);
            for (int p = 0; p < n; ++p) {
                const int cmb = order[static_cast<size_t>(p)];
                posc[ct.cards[cmb][0]].push_back(p);
                posc[ct.cards[cmb][1]].push_back(p);
            }
            auto pc = poscard.accessor<int64_t, 2>();
            for (int c = 0; c < kCards; ++c)
                for (int j = 0; j < 51; ++j)
                    pc[b][c * 51 + j] = posc[c][static_cast<size_t>(j)];
            auto la = ilowA.accessor<int64_t, 2>();
            auto lb2 = ilowB.accessor<int64_t, 2>();
            auto ea = iendA.accessor<int64_t, 2>();
            auto eb = iendB.accessor<int64_t, 2>();
            auto ta = itotA.accessor<int64_t, 2>();
            auto tb = itotB.accessor<int64_t, 2>();
            auto count_below = [&](int c, int64_t q) -> int64_t {
                const auto& v = posc[c];
                return std::lower_bound(v.begin(), v.end(),
                                        static_cast<int>(q)) -
                       v.begin();
            };
            for (int p = 0; p < n; ++p) {
                const int cmb = order[static_cast<size_t>(p)];
                const int A = ct.cards[cmb][0], Bc = ct.cards[cmb][1];
                la[b][p] = A * 52 + count_below(A, pe[b][p]);
                lb2[b][p] = Bc * 52 + count_below(Bc, pe[b][p]);
                ea[b][p] = A * 52 + count_below(A, se[b][p]);
                eb[b][p] = Bc * 52 + count_below(Bc, se[b][p]);
                ta[b][p] = A * 52 + 51;
                tb[b][p] = Bc * 52 + 51;
            }
        }

        auto cc0 = c0.accessor<float, 2>();
        auto cc1 = c1.accessor<float, 2>();
        for (int m = 0; m < M; ++m) {
            cc0[b][m] = static_cast<float>(sp.node_contrib0[m]);
            cc1[b][m] = static_cast<float>(sp.node_contrib1[m]);
        }
    }

    inc_ = inc.to(dev_).to(dt_);
    card_a_ = ca.to(dev_);
    card_b_ = cb.to(dev_);
    valid_ = valid.to(dev_).to(dt_);
    r0_ = (r0 * valid).to(dev_).to(dt_);
    r1_ = (r1 * valid).to(dev_).to(dt_);
    perm_ = perm.to(dev_);
    prev_end1_ = prev1.to(dev_);
    seg_end_ = segend.to(dev_);
    pos_card_ = poscard.to(dev_);
    idx_lowA_ = ilowA.to(dev_);
    idx_lowB_ = ilowB.to(dev_);
    idx_endA_ = iendA.to(dev_);
    idx_endB_ = iendB.to(dev_);
    idx_totA_ = itotA.to(dev_);
    idx_totB_ = itotB.to(dev_);
    c0_ = c0.to(dev_).to(dt_);
    c1_ = c1.to(dev_).to(dt_);

    regret_.resize(shape_.nodes.size());
    cum_.resize(shape_.nodes.size());
    for (size_t m = 0; m < shape_.nodes.size(); ++m) {
        if (shape_.nodes[m].kind != HunlSolver::Node::Decision) continue;
        const long A = static_cast<long>(shape_.nodes[m].acts.size());
        regret_[m] = torch::zeros(
            {B_, A, n},
            torch::TensorOptions().dtype(dt_).device(dev_));
        cum_[m] = torch::zeros_like(regret_[m]);
    }
    t_dev_ = torch::ones({}, torch::TensorOptions().dtype(dt_).device(dev_));
}

torch::Tensor BatchRiverSolver::policies(int node, bool average) {
    const auto& src = average ? cum_[node] : regret_[node];
    auto pos = torch::clamp_min(src, 0.0f);
    auto s = pos.sum(1, /*keepdim=*/true);                  // [B,1,n]
    const float uni =
        1.0f / static_cast<float>(shape_.nodes[node].acts.size());
    return torch::where(s > 1e-12f, pos / s.clamp_min(1e-12f),
                        torch::full_like(pos, uni));
}

torch::Tensor BatchRiverSolver::fold_cfv_t(int node, int upd,
                                           const torch::Tensor& opp) {
    // mass = S − Sc[a] − Sc[b] + w   (all [B, n])
    auto S = opp.sum(1, /*keepdim=*/true);
    auto Sc = torch::matmul(opp, inc_);                     // [B, 52]
    auto mass = S - Sc.index_select(1, card_a_) -
                Sc.index_select(1, card_b_) + opp;
    const int folder = shape_.nodes[node].player;
    auto u = (upd == folder)
        ? -(upd == 0 ? c0_ : c1_).select(1, node)
        : (upd == 0 ? c1_ : c0_).select(1, node);           // [B]
    return mass * u.unsqueeze(1) * valid_;
}

torch::Tensor BatchRiverSolver::showdown_cfv_t(int node,
                                               const torch::Tensor& opp) {
    const int n = kCombos;
    auto ws = opp.gather(1, perm_);                         // [B, n]
    auto cs = ws.cumsum(1);
    auto cs0 = torch::cat({torch::zeros({B_, 1}, ws.options()), cs}, 1);
    // card-restricted prefix masses over each card's own 51 positions —
    // [B, 52, 52] zero-padded table; boundary lookups are the precomputed
    // constant (card·52 + count) indices
    auto wsc = ws.gather(1, pos_card_).reshape({B_, kCards, 51});
    auto csc = wsc.cumsum(2);
    auto csc0 = torch::cat(
        {torch::zeros({B_, kCards, 1}, ws.options()), csc}, 2)
                    .reshape({B_, kCards * 52});

    auto lowT = cs0.gather(1, prev_end1_);
    auto lowA = csc0.gather(1, idx_lowA_);
    auto lowB = csc0.gather(1, idx_lowB_);
    auto win = lowT - lowA - lowB;

    auto endT = cs0.gather(1, seg_end_);
    auto endA = csc0.gather(1, idx_endA_);
    auto endB = csc0.gather(1, idx_endB_);
    auto tie = (endT - lowT) - (endA - lowA) - (endB - lowB) + ws;

    auto S = cs.select(1, n - 1).unsqueeze(1);              // [B,1]
    auto ScA = csc0.gather(1, idx_totA_);
    auto ScB = csc0.gather(1, idx_totB_);
    auto tot = S - ScA - ScB + ws;

    auto half_pot = c0_.select(1, node).unsqueeze(1);       // contribs equal
    auto cfv_sorted = half_pot * (2.0f * win + tie - tot);
    auto cfv = torch::zeros_like(cfv_sorted);
    cfv.scatter_(1, perm_, cfv_sorted);
    return cfv * valid_;
}

torch::Tensor BatchRiverSolver::walk(int node, int upd, int t, bool update,
                                     torch::Tensor my_reach,
                                     torch::Tensor opp_reach) {
    const auto& nd = shape_.nodes[node];
    switch (static_cast<HunlSolver::Node::Kind>(nd.kind)) {
    case HunlSolver::Node::Fold:
        return fold_cfv_t(node, upd, opp_reach);
    case HunlSolver::Node::Showdown:
    case HunlSolver::Node::AllinShowdown:   // river: board complete
        return showdown_cfv_t(node, opp_reach);
    case HunlSolver::Node::StreetEnd:
        TORCH_CHECK(false, "river batch solver has no StreetEnd leaves");
    case HunlSolver::Node::Decision:
        break;
    }

    const int A = static_cast<int>(nd.acts.size());
    auto sig = policies(node, /*average=*/!update);         // [B, A, n]
    if (nd.player == upd) {
        std::vector<torch::Tensor> cfv_a(A);
        for (int k = 0; k < A; ++k)
            cfv_a[k] = walk(nd.child[k], upd, t, update,
                            my_reach * sig.select(1, k), opp_reach);
        auto stacked = torch::stack(cfv_a, 1);              // [B, A, n]
        auto v = (sig * stacked).sum(1);                    // [B, n]
        if (update) {
            // strictly IN-PLACE on stable storage: a captured graph replays
            // exactly these reads/writes, so the recurrence (iteration N+1
            // reads what N wrote) only holds if the buffers never rebind.
            // add_/sub_ split keeps the ORIGINAL association ((r+s)−v):
            // RM⁺'s clamp is discontinuous, so a rounding-order change
            // flips clamp decisions and lands on a different (equally
            // valid) equilibrium selection — f64 equivalence vs the CPU
            // solver is only bitwise-stable with the original order.
            regret_[node].add_(stacked).sub_(v.unsqueeze(1)).clamp_min_(0.0);
            cum_[node].add_((t_dev_ * my_reach.unsqueeze(1)) * sig);
        }
        return v;
    }
    torch::Tensor cfv;
    for (int k = 0; k < A; ++k) {
        auto cv = walk(nd.child[k], upd, t, update, my_reach,
                       opp_reach * sig.select(1, k));
        cfv = cfv.defined() ? cfv + cv : cv;
    }
    return cfv;
}

void BatchRiverSolver::iterate(int t) {
    torch::NoGradGuard ng;
    t_dev_.fill_(static_cast<double>(t));
    for (int upd = 0; upd < 2; ++upd)
        walk(0, upd, t, /*update=*/true, upd == 0 ? r0_ : r1_,
             upd == 0 ? r1_ : r0_);
}

bool BatchRiverSolver::try_fused(int T) {
    const int M = static_cast<int>(shape_.nodes.size());
    // kernel compile-time bounds: stack depth < 12, arity ≤ 8
    std::vector<int> depth(static_cast<size_t>(M), 0);
    int maxd = 0, maxA = 0;
    for (int m = 0; m < M; ++m) {   // creation order: parents precede kids
        for (int c : shape_.nodes[m].child)
            depth[static_cast<size_t>(c)] = depth[static_cast<size_t>(m)] + 1;
        maxd = std::max(maxd, depth[static_cast<size_t>(m)]);
        maxA = std::max(maxA,
                        static_cast<int>(shape_.nodes[m].acts.size()));
    }
    if (maxd + 1 >= 12 || maxA > 8) return false;

    auto li = torch::TensorOptions().dtype(torch::kInt);
    auto ll = torch::TensorOptions().dtype(torch::kLong);
    auto kind = torch::empty({M}, li);
    auto act = torch::empty({M}, li);
    auto ar = torch::empty({M}, li);
    auto cbase = torch::empty({M}, li);
    std::vector<int> cflat;
    auto rptr = torch::zeros({M}, ll);
    auto cptr = torch::zeros({M}, ll);
    {
        auto ka = kind.accessor<int, 1>();
        auto aa = act.accessor<int, 1>();
        auto ra = ar.accessor<int, 1>();
        auto ba = cbase.accessor<int, 1>();
        auto rp = rptr.accessor<int64_t, 1>();
        auto cp = cptr.accessor<int64_t, 1>();
        for (int m = 0; m < M; ++m) {
            const auto& nd = shape_.nodes[m];
            ka[m] = nd.kind;
            aa[m] = nd.player;
            ra[m] = static_cast<int>(nd.acts.size());
            ba[m] = static_cast<int>(cflat.size());
            for (int c : nd.child) cflat.push_back(c);
            if (regret_[static_cast<size_t>(m)].defined()) {
                rp[m] = reinterpret_cast<int64_t>(
                    regret_[static_cast<size_t>(m)].data_ptr<float>());
                cp[m] = reinterpret_cast<int64_t>(
                    cum_[static_cast<size_t>(m)].data_ptr<float>());
            }
        }
    }
    auto cflat_t = torch::tensor(cflat, li);
    // the address/topology tables must live on-device for the kernel
    auto kind_d = kind.to(dev_);
    auto act_d = act.to(dev_);
    auto ar_d = ar.to(dev_);
    auto cbase_d = cbase.to(dev_);
    auto cflat_d = cflat_t.to(dev_);
    auto rptr_d = rptr.to(dev_);
    auto cptr_d = cptr.to(dev_);

    FusedRiverArgs a;
    a.T = T;
    a.M = M;
    a.B = B_;
    a.kind = kind_d.data_ptr<int>();
    a.actor = act_d.data_ptr<int>();
    a.arity = ar_d.data_ptr<int>();
    a.child_base = cbase_d.data_ptr<int>();
    a.child_flat = cflat_d.data_ptr<int>();
    a.regret_ptr = rptr_d.data_ptr<int64_t>();
    a.cum_ptr = cptr_d.data_ptr<int64_t>();
    a.r0 = r0_.data_ptr<float>();
    a.r1 = r1_.data_ptr<float>();
    a.valid = valid_.data_ptr<float>();
    a.c0 = c0_.data_ptr<float>();
    a.c1 = c1_.data_ptr<float>();
    a.perm = perm_.data_ptr<int64_t>();
    a.prev1 = prev_end1_.data_ptr<int64_t>();
    a.segend = seg_end_.data_ptr<int64_t>();
    a.poscard = pos_card_.data_ptr<int64_t>();
    a.ilowA = idx_lowA_.data_ptr<int64_t>();
    a.ilowB = idx_lowB_.data_ptr<int64_t>();
    a.iendA = idx_endA_.data_ptr<int64_t>();
    a.iendB = idx_endB_.data_ptr<int64_t>();
    a.itotA = idx_totA_.data_ptr<int64_t>();
    a.itotB = idx_totB_.data_ptr<int64_t>();
    a.cardA = card_a_.data_ptr<int64_t>();
    a.cardB = card_b_.data_ptr<int64_t>();
    torch::cuda::synchronize();   // context current, uploads complete
    return fused_river_solve(a);
}

void BatchRiverSolver::solve(int T) {
    torch::NoGradGuard ng;
    const bool no_graph = std::getenv("REBEL_NO_CUDA_GRAPH") != nullptr;
    if (!dev_.is_cuda() || T < 8 || no_graph) {
        for (int t = 1; t <= T; ++t) iterate(t);
        return;
    }
    // persistent NVRTC kernel: the whole T-iteration solve in ONE launch
    if (dt_ == torch::kFloat && !std::getenv("REBEL_NO_FUSED") &&
        try_fused(T))
        return;
    // warmup (allocator + kernel caches settle), then capture the whole
    // two-pass iteration; per remaining iteration: one fill_ + one replay
    try {
        auto stream = at::cuda::getStreamFromPool();
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            for (int t = 1; t <= 3; ++t) iterate(t);
            at::cuda::getCurrentCUDAStream().synchronize();
            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            for (int upd = 0; upd < 2; ++upd)
                walk(0, upd, /*t=*/0, /*update=*/true,
                     upd == 0 ? r0_ : r1_, upd == 0 ? r1_ : r0_);
            graph_->capture_end();
            at::cuda::getCurrentCUDAStream().synchronize();
        }
        // capture RECORDS without executing: iteration 4 onward comes
        // entirely from replays
        for (int t = 4; t <= T; ++t) {
            t_dev_.fill_(static_cast<double>(t));
            graph_->replay();
        }
        torch::cuda::synchronize();
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[gpu] CUDA graph capture failed (%s) — eager "
                     "fallback\n", e.what());
        graph_.reset();
        for (int t = 4; t <= T; ++t) iterate(t);
    }
}

void BatchRiverSolver::root_values(
    std::vector<std::array<std::vector<double>, 2>>& v,
    std::vector<std::array<std::vector<double>, 2>>& mask) {
    torch::NoGradGuard ng;
    v.assign(static_cast<size_t>(B_), {});
    mask.assign(static_cast<size_t>(B_), {});
    for (int p = 0; p < 2; ++p) {
        auto own = p == 0 ? r0_ : r1_;
        auto opp = p == 0 ? r1_ : r0_;
        auto cfv = walk(0, p, /*t=*/0, /*update=*/false, own, opp);
        // opponent compatible mass
        auto S = opp.sum(1, true);
        auto Sc = torch::matmul(opp, inc_);
        auto m = (S - Sc.index_select(1, card_a_) -
                  Sc.index_select(1, card_b_) + opp) * valid_;
        auto vals = torch::where(m > 1e-6, cfv / m.clamp_min(1e-6),
                                 torch::zeros_like(cfv))
                        .to(torch::kCPU).to(torch::kDouble).contiguous();
        auto ok = ((m > 1e-6) & (own > 0.0))
                      .to(torch::kDouble).to(torch::kCPU).contiguous();
        auto va = vals.accessor<double, 2>();
        auto oa = ok.accessor<double, 2>();
        for (int b = 0; b < B_; ++b) {
            v[static_cast<size_t>(b)][p].assign(kCombos, 0.0);
            mask[static_cast<size_t>(b)][p].assign(kCombos, 0.0);
            for (int i = 0; i < kCombos; ++i) {
                v[static_cast<size_t>(b)][p][i] = va[b][i];
                mask[static_cast<size_t>(b)][p][i] = oa[b][i];
            }
        }
    }
}

}  // namespace rebel_hunl
