#include "checkpoint.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <vector>

namespace poker_ppo {

namespace fs = std::filesystem;

namespace {

// Parses "update_<N>.pt" -> N, or -1 if not in that form.
int parse_update_n(const std::string& filename) {
    static const std::regex re(R"(update_(\d+)\.pt)");
    std::smatch m;
    if (std::regex_match(filename, m, re)) {
        try { return std::stoi(m[1]); } catch (...) { return -1; }
    }
    return -1;
}

// Returns (n, path) sorted ascending by n.
std::vector<std::pair<int, fs::path>>
list_checkpoints(const fs::path& dir) {
    std::vector<std::pair<int, fs::path>> out;
    if (!fs::exists(dir)) return out;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const int n = parse_update_n(entry.path().filename().string());
        if (n >= 0) out.emplace_back(n, entry.path());
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

}  // namespace

Checkpoint::Checkpoint(fs::path dir) : dir_(std::move(dir)) {
    fs::create_directories(dir_);
}

void Checkpoint::save(int update_idx, int global_step,
                      ActorCritic& network,
                      torch::optim::Adam& optimizer,
                      ActorCritic& magnet)
{
    const auto path = dir_ / ("update_" + std::to_string(update_idx) + ".pt");

    // Write to a temp path then rename — protects against half-written
    // files if the process dies mid-save (atomic on POSIX/NTFS).
    const auto tmp_path = dir_ / ("update_" + std::to_string(update_idx) + ".pt.tmp");

    {
        torch::serialize::OutputArchive archive;

        archive.write("update_idx",
                      torch::tensor(static_cast<int64_t>(update_idx), torch::kInt64));
        archive.write("global_step",
                      torch::tensor(static_cast<int64_t>(global_step), torch::kInt64));
        archive.write("has_magnet",
                      torch::tensor(magnet.is_empty() ? 0 : 1, torch::kInt32));

        {
            torch::serialize::OutputArchive sub;
            network->save(sub);
            archive.write("network", sub);
        }
        {
            torch::serialize::OutputArchive sub;
            optimizer.save(sub);
            archive.write("optimizer", sub);
        }
        if (!magnet.is_empty()) {
            torch::serialize::OutputArchive sub;
            magnet->save(sub);
            archive.write("magnet", sub);
        }

        archive.save_to(tmp_path.string());
    }

    fs::rename(tmp_path, path);
    std::cout << "[ckpt] saved update=" << update_idx
              << " step=" << global_step
              << " -> " << path.filename().string() << "\n";
}

std::optional<Checkpoint::LoadedState>
Checkpoint::load_latest(ActorCritic& network,
                        torch::optim::Adam& optimizer,
                        ActorCritic& magnet)
{
    const auto ckpts = list_checkpoints(dir_);
    if (ckpts.empty()) return std::nullopt;

    const auto& [n, path] = ckpts.back();

    torch::serialize::InputArchive archive;
    archive.load_from(path.string());

    torch::Tensor uidx_t, gstep_t, has_mag_t;
    archive.read("update_idx", uidx_t);
    archive.read("global_step", gstep_t);
    archive.read("has_magnet", has_mag_t);

    LoadedState state{};
    state.update_idx     = static_cast<int>(uidx_t.item<int64_t>());
    state.global_step    = static_cast<int>(gstep_t.item<int64_t>());
    state.magnet_present = has_mag_t.item<int>() != 0;

    {
        torch::serialize::InputArchive sub;
        archive.read("network", sub);
        network->load(sub);
    }
    {
        torch::serialize::InputArchive sub;
        archive.read("optimizer", sub);
        optimizer.load(sub);
    }
    if (state.magnet_present && !magnet.is_empty()) {
        torch::serialize::InputArchive sub;
        archive.read("magnet", sub);
        magnet->load(sub);
    }

    std::cout << "[ckpt] resumed from " << path.filename().string()
              << " (update=" << state.update_idx
              << " step=" << state.global_step << ")\n";
    return state;
}

void Checkpoint::prune(int keep_last) {
    if (keep_last <= 0) return;
    const auto ckpts = list_checkpoints(dir_);
    if (static_cast<int>(ckpts.size()) <= keep_last) return;

    const int to_delete = static_cast<int>(ckpts.size()) - keep_last;
    for (int i = 0; i < to_delete; ++i) {
        std::error_code ec;
        fs::remove(ckpts[i].second, ec);
        if (ec) {
            std::cerr << "[ckpt] failed to delete " << ckpts[i].second
                      << ": " << ec.message() << "\n";
        }
    }
}

}  // namespace poker_ppo
