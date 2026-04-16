cmake_minimum_required(VERSION 3.18)
project(poker_ppo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ──────────────────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(poker_ppo_lib
    src/network.cpp
    src/ppo.cpp
    src/rollout_buffer.cpp
)
target_include_directories(poker_ppo_lib PUBLIC include)
target_link_libraries(poker_ppo_lib PUBLIC "${TORCH_LIBRARIES}")

# ── Example binary ────────────────────────────────────────────────────────────
add_executable(poker_ppo src/main.cpp)
target_link_libraries(poker_ppo PRIVATE poker_ppo_lib)
