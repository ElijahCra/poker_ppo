#include "hunl_gpu.h"

#include <algorithm>
#include <cstdio>

namespace rebel_hunl {

namespace {
constexpr double kTiny = 1e-12;
}

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
    auto asort = torch::zeros({B_, n}, l);
    auto bsort = torch::zeros({B_, n}, l);
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
        auto as = asort.accessor<int64_t, 2>();
        auto bs = bsort.accessor<int64_t, 2>();
        for (int i = 0; i < n; ++i) {
            va[b][i] = vb[i] ? 1.0f : 0.0f;
            if (vb[i]) {
                a0[b][i] = s0 > kTiny ? static_cast<float>(sp.r0[i] / s0)
                                      : static_cast<float>(1.0 / nv);
                a1[b][i] = s1 > kTiny ? static_cast<float>(sp.r1[i] / s1)
                                      : static_cast<float>(1.0 / nv);
            }
        }
        // segment boundaries over the sorted VALID prefix; pads form one
        // trailing zero-mass segment.
        int gstart = 0;
        for (int pos = 0; pos < n; ++pos) {
            const int cmb = order[static_cast<size_t>(pos)];
            pp[b][pos] = cmb;
            as[b][pos] = ct.cards[cmb][0];
            bs[b][pos] = ct.cards[cmb][1];
            const bool last_of_group =
                pos + 1 >= nv || pos + 1 >= n ||
                (pos + 1 < nv &&
                 rank[order[static_cast<size_t>(pos + 1)]] != rank[cmb]);
            (void)last_of_group;
        }
        // second pass for group indices (clearer than fusing above)
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
    a_sorted_ = asort.to(dev_);
    b_sorted_ = bsort.to(dev_);
    card_sorted_ = inc_.index_select(0, perm_.reshape({-1}))
                       .reshape({B_, n, kCards});
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
    // per-card cumulative masses along the sorted order
    auto wc = (ws.unsqueeze(2) * card_sorted_).cumsum(1);
    auto wc0 = torch::cat(
        {torch::zeros({B_, 1, kCards}, ws.options()), wc}, 1);

    auto gather_tot = [&](const torch::Tensor& idx) {       // [B,n] from cs0
        return cs0.gather(1, idx);
    };
    auto gather_card = [&](const torch::Tensor& idx,
                           const torch::Tensor& cards) {    // [B,n]
        auto t = wc0.gather(
            1, idx.unsqueeze(2).expand({B_, n, kCards}));   // [B,n,52]
        return t.gather(2, cards.unsqueeze(2)).squeeze(2);
    };

    auto lowT = gather_tot(prev_end1_);
    auto lowA = gather_card(prev_end1_, a_sorted_);
    auto lowB = gather_card(prev_end1_, b_sorted_);
    auto win = lowT - lowA - lowB;

    auto endT = gather_tot(seg_end_);
    auto endA = gather_card(seg_end_, a_sorted_);
    auto endB = gather_card(seg_end_, b_sorted_);
    auto tie = (endT - lowT) - (endA - lowA) - (endB - lowB) + ws;

    auto S = cs.select(1, n - 1).unsqueeze(1);              // [B,1]
    auto ScA = gather_card(torch::full_like(prev_end1_, n), a_sorted_);
    auto ScB = gather_card(torch::full_like(prev_end1_, n), b_sorted_);
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
            regret_[node] =
                torch::clamp_min(regret_[node] + stacked - v.unsqueeze(1),
                                 0.0f);
            cum_[node] = cum_[node] +
                static_cast<float>(t) * my_reach.unsqueeze(1) * sig;
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
    for (int upd = 0; upd < 2; ++upd)
        walk(0, upd, t, /*update=*/true, upd == 0 ? r0_ : r1_,
             upd == 0 ? r1_ : r0_);
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
