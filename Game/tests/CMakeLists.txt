# Showdown / hand-evaluator unit tests.
#
# Pulls GoogleTest via FetchContent so this is self-contained — no system
# install required. Run with:
#     cmake --build cmake-build-release --target showdown_test
#     ./cmake-build-release/Game/tests/showdown_test
# or via ctest:
#     cd cmake-build-release && ctest --output-on-failure -R showdown

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
# Prevent GoogleTest from overriding our compiler/linker options on Windows;
# harmless on other platforms.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(showdown_test showdown_test.cpp)

# Utility (TwoPlusTwo evaluator) is the only Game lib piece needed; the
# tests don't pull in DiscreteGame / Transitioner so they stay focused on
# the hand-strength path that the bug fix landed on.
target_link_libraries(showdown_test PRIVATE
    Game            # interface lib that exposes Game/ as include root
    Utility         # static lib with the TwoPlusTwo evaluator implementation
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# Attention-encoder tests. These exercise the network module so they need
# poker_ppo_lib (which transitively pulls in libtorch).
add_executable(attention_test attention_test.cpp)
target_link_libraries(attention_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# BettingConfig tests. Header-only path (BettingConfig.hpp + GameConfig.hpp
# + GameBase.hpp), so we just need the Game interface lib for include
# directories. DiscreteGame's transition path pulls in Utility for the
# TwoPlusTwo evaluator.
add_executable(betting_config_test betting_config_test.cpp)
target_link_libraries(betting_config_test PRIVATE
    Game
    Utility
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# PokerConfig tests. Pulls poker_env.h (transitively config.h + Game) so
# we link against poker_ppo_lib for the include path + libtorch.
add_executable(poker_config_test poker_config_test.cpp)
target_link_libraries(poker_config_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# HandStrength tests. Pure header-only helper — no dependencies beyond
# the Game include root, so we just need the interface lib here.
add_executable(hand_strength_test hand_strength_test.cpp)
target_link_libraries(hand_strength_test PRIVATE
    Game
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# MMD tests. Exercises ActorCritic::masked_log_probs and the KL math
# the trainer uses; needs poker_ppo_lib for the network module + libtorch.
add_executable(mmd_test mmd_test.cpp)
target_link_libraries(mmd_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# All-in equity tests. Header-only helper over the TwoPlusTwo evaluator, so
# only the Game include root + Utility (evaluator impl) are needed.
add_executable(allin_equity_test allin_equity_test.cpp)
target_link_libraries(allin_equity_test PRIVATE
    Game
    Utility
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# VRPO advantage tests. Exercise RolloutBuffer::compute_returns (the unified
# GAE / Expected-SARSA(λ) trace), so link poker_ppo_lib for the buffer.
add_executable(vrpo_advantage_test vrpo_advantage_test.cpp)
target_link_libraries(vrpo_advantage_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# ForeachAdam / foreach_clip_grad_norm equivalence vs torch::optim::Adam +
# torch::nn::utils::clip_grad_norm_. Needs poker_ppo_lib for optim.cpp.
add_executable(optim_test optim_test.cpp)
target_link_libraries(optim_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# History-encoder variants (attn/conv/pool/flatten): shapes, masking
# semantics, ActorCritic integration + clone.
add_executable(history_encoder_test history_encoder_test.cpp)
target_link_libraries(history_encoder_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
    GTest::gtest      # explicit: some toolchains (strict --as-needed / link
                      # order) don't pull gtest_main's transitive gtest dep
)

# Register with CTest so `ctest` discovers the cases automatically.
include(GoogleTest)
gtest_discover_tests(showdown_test)
gtest_discover_tests(attention_test)
gtest_discover_tests(betting_config_test)
gtest_discover_tests(poker_config_test)
gtest_discover_tests(hand_strength_test)
gtest_discover_tests(mmd_test)
gtest_discover_tests(allin_equity_test)
gtest_discover_tests(vrpo_advantage_test)
gtest_discover_tests(optim_test)
gtest_discover_tests(history_encoder_test)
