#include "commands.h"

#include "poker_env.h"

#include <iostream>
#include <sstream>
#include <string>

namespace poker_ppo {

// cmd_play — interactive REPL exposing one PokerEnvironment + the trained
// network over stdin/stdout. Driven by tools/play.py for the UI.
//
// Protocol: each command from the client is a single line. The server
// responds with multiple `key value...` lines terminated by an `OK` line.
// All amounts are in mbb; cards are integer ids (rank<<2 | suit) in [0, 52).
//
// Commands:
//   INFO                — emit static config (action_count, obs_dim,
//                         blinds, pot_fractions, etc.).
//   STATE               — emit current game state (player, cards, stacks,
//                         pot, mask, obs, done, utility).
//   STEP <action_idx>   — apply the given action; emit new STATE.
//   RESET               — reset to a fresh hand; emit new STATE.
//   MODEL               — run the trained network on the current obs; emit
//                         chosen action + full softmax probs + value.
//   QUIT                — exit cleanly.
//
// Errors: emit a single `ERR <message>` line followed by `OK`.

int cmd_play(IPokerEnvironmentFactory& factory,
             const BetConfig&          bet_cfg,
             torch::Device             device,
             const std::string&        model_path)
{
    if (model_path.empty()) {
        std::cerr << "--play requires a model path: ./poker_ppo --play <path>\n";
        return 1;
    }
    std::cerr << "[play] loading " << model_path << "\n";

    PPOTrainer trainer(factory, device);
    trainer.load(model_path);
    auto& net = trainer.network();
    net->eval();

    // Single env for play. Need the concrete PokerEnvironment for the
    // state-inspection accessors (hole_cards, pot, etc.).
    auto env_base = factory.create(bet_cfg);
    auto* env = dynamic_cast<PokerEnvironment*>(env_base.get());
    if (!env) {
        std::cerr << "ERR factory did not produce PokerEnvironment\n";
        return 1;
    }
    env->reset();

    auto emit_info = [&]() {
        const auto& g = env->game_config();
        std::cout << "obs_dim "      << env->obs_dim()             << "\n";
        std::cout << "action_count " << bet_cfg.action_count()     << "\n";
        std::cout << "initial_stack " << g.initial_stack           << "\n";
        std::cout << "small_blind "   << g.small_blind             << "\n";
        std::cout << "big_blind "     << g.big_blind               << "\n";
        std::cout << "min_raise "     << g.min_raise               << "\n";
        std::cout << "max_raises_per_round " << int(g.max_raises_per_round) << "\n";
        std::cout << "pot_fractions " << g.pot_fractions.size();
        for (double f : g.pot_fractions) std::cout << " " << f;
        std::cout << "\n";
        std::cout << "has_allin " << (g.include_all_in_slot ? 1 : 0) << "\n";
        std::cout << "OK" << std::endl;
    };

    auto emit_state = [&]() {
        std::cout << "cur_player " << env->current_player() << "\n";
        std::cout << "round "      << env->round()          << "\n";
        std::cout << "pot "        << env->pot()            << "\n";
        std::cout << "cb "         << env->current_bet()    << "\n";
        std::cout << "raises "     << env->raise_num()      << "\n";
        const auto h0 = env->hole_cards(0);
        const auto h1 = env->hole_cards(1);
        std::cout << "hole_p0 " << h0[0] << " " << h0[1] << "\n";
        std::cout << "hole_p1 " << h1[0] << " " << h1[1] << "\n";
        std::cout << "stacks "  << env->stack(0) << " " << env->stack(1) << "\n";
        const int n_comm = env->community_count();
        std::cout << "board " << n_comm;
        for (int i = 0; i < n_comm; ++i)
            std::cout << " " << env->community_card(i);
        std::cout << "\n";
        const auto mask  = env->legal_action_mask();
        const auto m_acc = mask.accessor<float, 1>();
        std::cout << "mask " << mask.size(0);
        for (int i = 0; i < mask.size(0); ++i)
            std::cout << " " << (m_acc[i] > 0.5f ? 1 : 0);
        std::cout << "\n";
        const auto obs   = env->observation();
        const auto o_acc = obs.accessor<float, 1>();
        std::cout << "obs " << obs.size(0);
        for (int i = 0; i < obs.size(0); ++i)
            std::cout << " " << o_acc[i];
        std::cout << "\n";
        std::cout << "done " << (env->is_terminal() ? 1 : 0) << "\n";
        if (env->is_terminal()) {
            std::cout << "utility_p0 " << env->terminal_utility(0) << "\n";
            std::cout << "utility_p1 " << env->terminal_utility(1) << "\n";
        }
        std::cout << "OK" << std::endl;
    };

    auto emit_model = [&]() {
        torch::NoGradGuard ng;
        auto obs  = env->observation().unsqueeze(0).to(device);
        auto mask = env->legal_action_mask().unsqueeze(0).to(device);
        auto [logits, value] = net->forward(obs);
        const auto masked = logits + (1.0f - mask) * kIllegalActionLogit;
        const auto probs  = torch::softmax(masked, -1).squeeze(0).to(torch::kCPU).contiguous();
        // Sample one action stochastically (matches training rollout policy).
        const auto ar      = net->get_action(obs, mask);
        const auto sampled = ar.action.to(torch::kCPU).item<int64_t>();
        // Greedy (argmax) action for diagnostics.
        const auto greedy  = std::get<1>(probs.max(0)).item<int64_t>();
        std::cout << "sampled "  << sampled << "\n";
        std::cout << "greedy "   << greedy  << "\n";
        std::cout << "value "    << value.squeeze(0).item<float>() << "\n";
        const auto p_acc = probs.accessor<float, 1>();
        std::cout << "probs " << probs.size(0);
        for (int i = 0; i < probs.size(0); ++i)
            std::cout << " " << p_acc[i];
        std::cout << "\n";
        std::cout << "OK" << std::endl;
    };

    // Ready signal so the client can sync.
    std::cout << "READY" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        try {
            if (cmd == "INFO") {
                emit_info();
            } else if (cmd == "STATE") {
                emit_state();
            } else if (cmd == "STEP") {
                int action;
                if (!(iss >> action)) {
                    std::cout << "ERR bad_step_arg\nOK" << std::endl;
                    continue;
                }
                env->step(action);
                emit_state();
            } else if (cmd == "RESET") {
                env->reset();
                emit_state();
            } else if (cmd == "MODEL") {
                emit_model();
            } else if (cmd == "QUIT") {
                break;
            } else {
                std::cout << "ERR unknown_cmd\nOK" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "ERR " << e.what() << "\nOK" << std::endl;
        }
    }
    return 0;
}

} // namespace poker_ppo
