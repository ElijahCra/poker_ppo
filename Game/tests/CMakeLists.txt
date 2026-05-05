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
)

# Attention-encoder tests. These exercise the network module so they need
# poker_ppo_lib (which transitively pulls in libtorch).
add_executable(attention_test attention_test.cpp)
target_link_libraries(attention_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
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
)

# PokerConfig tests. Pulls poker_env.h (transitively config.h + Game) so
# we link against poker_ppo_lib for the include path + libtorch.
add_executable(poker_config_test poker_config_test.cpp)
target_link_libraries(poker_config_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
)

# HandStrength tests. Pure header-only helper — no dependencies beyond
# the Game include root, so we just need the interface lib here.
add_executable(hand_strength_test hand_strength_test.cpp)
target_link_libraries(hand_strength_test PRIVATE
    Game
    GTest::gtest_main
)

# MMD tests. Exercises ActorCritic::masked_log_probs and the KL math
# the trainer uses; needs poker_ppo_lib for the network module + libtorch.
add_executable(mmd_test mmd_test.cpp)
target_link_libraries(mmd_test PRIVATE
    poker_ppo_lib
    GTest::gtest_main
)

# Register with CTest so `ctest` discovers the cases automatically.
include(GoogleTest)
gtest_discover_tests(showdown_test)
gtest_discover_tests(attention_test)
gtest_discover_tests(betting_config_test)
gtest_discover_tests(poker_config_test)
gtest_discover_tests(hand_strength_test)
gtest_discover_tests(mmd_test)
