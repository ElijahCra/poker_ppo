// Tiny imperfect-information games (Kuhn, N-card Kuhn, Leduc) with EXACT
// exploitability — the C++ port of research/escher/games.py, for validating
// the C++ neural ESCHER on the SAME small games before any HUNL wiring.
//
// Faithful line-for-line port of the validated Python. Histories are strings
// and the games are tiny, so this favours obvious correctness over speed.
#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace escher {

// cards = {p0, p1, public}; `public` unused for Kuhn.
using Cards = std::array<int, 3>;

struct Deal {
    Cards cards;
    double prob;
};

// Abstract extensive-form interface (mirrors the Python duck-typed game).
struct Game {
    virtual ~Game() = default;
    virtual std::string name() const = 0;
    virtual int n_actions() const = 0;
    virtual std::vector<Deal> deals() const = 0;
    virtual bool is_terminal(const std::string& h) const = 0;
    virtual int current_player(const std::string& h) const = 0;
    virtual std::vector<int> legal_actions(const std::string& h) const = 0;
    virtual std::string step(const std::string& h, int a) const = 0;
    virtual std::string infoset_key(const std::string& h,
                                    const Cards& cards) const = 0;
    virtual double terminal_util_p0(const std::string& h,
                                    const Cards& cards) const = 0;
};

// ─── Kuhn poker ─────────────────────────────────────────────────────────────
// 3 cards {0,1,2}, one each, 1-chip ante. Actions: 0='p' pass/check, 1='b'
// bet/call. Higher card wins at showdown.
class Kuhn : public Game {
public:
    static constexpr char ACT[2] = {'p', 'b'};

    std::string name() const override { return "kuhn"; }
    int n_actions() const override { return 2; }

    std::vector<Deal> deals() const override {
        std::vector<Deal> out;
        for (int a = 0; a < n_cards(); ++a)
            for (int b = 0; b < n_cards(); ++b)
                if (a != b) out.push_back({{a, b, -1}, 0.0});
        double p = 1.0 / static_cast<double>(out.size());
        for (auto& d : out) d.prob = p;
        return out;
    }

    bool is_terminal(const std::string& h) const override {
        return h == "pp" || h == "bp" || h == "bb" || h == "pbp" || h == "pbb";
    }
    int current_player(const std::string& h) const override {
        return static_cast<int>(h.size()) % 2;
    }
    std::vector<int> legal_actions(const std::string&) const override {
        return {0, 1};
    }
    std::string step(const std::string& h, int a) const override {
        return h + ACT[a];
    }
    std::string infoset_key(const std::string& h,
                            const Cards& cards) const override {
        return std::to_string(cards[current_player(h)]) + "|" + h;
    }
    double terminal_util_p0(const std::string& h,
                            const Cards& cards) const override {
        int p0 = cards[0], p1 = cards[1];
        int p0_wins = (p0 > p1) ? 1 : -1;
        if (h == "pp") return p0_wins;
        if (h == "bp") return 1;
        if (h == "pbp") return -1;
        if (h == "bb" || h == "pbb") return 2 * p0_wins;
        return 0;  // unreachable
    }

protected:
    virtual int n_cards() const { return 3; }
};

// N-card Kuhn: same single-round tree, N>3 ranks — a NN must generalise over
// card value (a trivially-correct larger testbed).
class KuhnN : public Kuhn {
public:
    explicit KuhnN(int n) : n_(n) {}
    std::string name() const override { return "kuhn" + std::to_string(n_); }
protected:
    int n_cards() const override { return n_; }
private:
    int n_;
};

// ─── Leduc poker (simplified) ───────────────────────────────────────────────
// 6-card deck = two suits of {0,1,2}. Each player 1 private card; one public
// card after round 1. Ante 1; bet sizes 2 (round 1) / 4 (round 2); max 2 raises
// per round. Pair-with-public beats high card.
class Leduc : public Game {
public:
    static constexpr char ACT[3] = {'f', 'c', 'r'};  // fold, call/check, raise
    static constexpr int RANKS = 3;

    std::string name() const override { return "leduc"; }
    int n_actions() const override { return 3; }

    std::vector<Deal> deals() const override {
        static const int deck[6] = {0, 0, 1, 1, 2, 2};
        // aggregate identical (rank) triples by probability
        std::vector<Deal> out;
        int total = 0;
        // first count total
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) {
                if (j == i) continue;
                for (int k = 0; k < 6; ++k)
                    if (k != i && k != j) ++total;
            }
        // accumulate
        std::vector<Deal> raw;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) {
                if (j == i) continue;
                for (int k = 0; k < 6; ++k) {
                    if (k == i || k == j) continue;
                    Cards c{deck[i], deck[j], deck[k]};
                    bool found = false;
                    for (auto& d : raw)
                        if (d.cards == c) { d.prob += 1.0 / total; found = true; break; }
                    if (!found) raw.push_back({c, 1.0 / total});
                }
            }
        return raw;
    }

    static bool round_over(const std::string& r) {
        if (!r.empty() && r.back() == 'f') return true;
        if (r == "cc") return true;
        if (!r.empty() && r.back() == 'c' &&
            r.find('r') != std::string::npos) return true;
        return false;
    }

    static std::string cur_round(const std::string& h) {
        auto pos = h.rfind('/');
        return pos == std::string::npos ? h : h.substr(pos + 1);
    }

    bool is_terminal(const std::string& h) const override {
        if (!h.empty() && h.back() == 'f') return true;
        auto pos = h.find('/');
        if (pos != std::string::npos) {  // two rounds present
            std::string r2 = h.substr(pos + 1);
            if (round_over(r2)) return true;
        }
        return false;
    }

    int current_player(const std::string& h) const override {
        return static_cast<int>(cur_round(h).size()) % 2;
    }

    static int n_raises(const std::string& r) {
        int c = 0;
        for (char ch : r) if (ch == 'r') ++c;
        return c;
    }

    std::vector<int> legal_actions(const std::string& h) const override {
        std::string r = cur_round(h);
        std::vector<int> acts = {1};  // call/check always legal
        bool facing_bet = !r.empty() && r.back() == 'r';
        if (facing_bet) acts.push_back(0);
        if (n_raises(r) < 2) acts.push_back(2);
        std::sort(acts.begin(), acts.end());
        return acts;
    }

    std::string step(const std::string& h, int a) const override {
        std::string nh = h + ACT[a];
        // advance to round 2 when round 1 closes by call/check (not fold)
        if (nh.find('/') == std::string::npos && round_over(nh) &&
            nh.back() != 'f')
            nh += '/';
        return nh;
    }

    std::string infoset_key(const std::string& h,
                            const Cards& cards) const override {
        int p = current_player(h);
        std::string pub = (h.find('/') != std::string::npos)
                              ? std::to_string(cards[2]) : "?";
        return std::to_string(cards[p]) + "|" + pub + "|" + h;
    }

    double terminal_util_p0(const std::string& h,
                            const Cards& cards) const override {
        int p0 = cards[0], p1 = cards[1], pub = cards[2];
        std::array<int, 2> contrib = {1, 1};
        // split rounds
        std::vector<std::string> rounds;
        {
            size_t start = 0, pos;
            while ((pos = h.find('/', start)) != std::string::npos) {
                rounds.push_back(h.substr(start, pos - start));
                start = pos + 1;
            }
            rounds.push_back(h.substr(start));
        }
        for (size_t ri = 0; ri < rounds.size(); ++ri) {
            int size = (ri == 0) ? 2 : 4;
            std::array<int, 2> cur = {0, 0};
            int to_call = 0, actor = 0;
            for (char ch : rounds[ri]) {
                if (ch == 'c') cur[actor] += to_call;
                else if (ch == 'r') { cur[actor] += to_call + size; to_call = size; }
                actor ^= 1;
            }
            contrib[0] += cur[0];
            contrib[1] += cur[1];
        }
        if (!h.empty() && h.back() == 'f') {
            std::string r = cur_round(h);
            int folder = (static_cast<int>(r.size()) - 1) % 2;
            int winner = 1 - folder;
            return winner == 0 ? contrib[1 - winner]
                               : -static_cast<double>(contrib[1 - winner]);
        }
        // showdown
        auto rank = [&](int card) { return card == pub ? 10 + card : card; };
        int r0 = rank(p0), r1 = rank(p1);
        if (r0 == r1) return 0.0;
        int winner = (r0 > r1) ? 0 : 1;
        return winner == 0 ? contrib[1] : -static_cast<double>(contrib[0]);
    }
};

}  // namespace escher
