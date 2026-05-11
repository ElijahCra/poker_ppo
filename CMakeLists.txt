cmake_minimum_required(VERSION 3.18)
project(poker_ppo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(Game)
# ── LibTorch ──────────────────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/share/cmake")
foreach(PY_CANDIDATE python python3 python3.12 python3.13 python3.14)
    execute_process(
            COMMAND ${PY_CANDIDATE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            RESULT_VARIABLE TORCH_FIND_RESULT
            ERROR_QUIET
    )
    if(TORCH_FIND_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
        message(STATUS "Found PyTorch via ${PY_CANDIDATE}: ${TORCH_CMAKE_PREFIX}")
        break()
    endif()
endforeach()
if(NOT TORCH_CMAKE_PREFIX)
    message(FATAL_ERROR "Could not locate PyTorch via Python. Make sure torch is installed in your active Python environment.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
# Point CMake to nvcc if not already in PATH (WSL CUDA toolkit location)
if(NOT CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc
            PATHS /usr/local/cuda/bin /usr/local/cuda-12.1/bin /usr/local/cuda-12/bin
    )
endif()
find_package(Torch REQUIRED)

# ── PPO library ───────────────────────────────────────────────────────────────
add_library(poker_ppo_lib
    src/network.cpp
    src/ppo.cpp
    src/rollout.cpp
    src/poker_env.cpp
    src/observation_builder.cpp
    src/league.cpp
    src/metrics_logger.cpp
    src/opponent_manager.cpp
    src/best_response.cpp
    src/commands.cpp
)
target_include_directories(poker_ppo_lib PUBLIC
    include
)
target_link_libraries(poker_ppo_lib PUBLIC
    "${TORCH_LIBRARIES}"
    Game
)

# libtorch ≥ 2.8 references free / memcpy / numeric_limits<> / lock_guard<>
# in c10/ATen headers without including <cstdlib>/<cstring>/<limits>/<mutex>,
# relying on transitive includes that aren't guaranteed. Force-include them
# here. (`features.h` first so glibc feature-test macros are defined before
# any libstdc++ header that gates on them.)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # libtorch ≥ 2.8 references free / memcpy / numeric_limits<> / lock_guard<>
    # in c10/ATen headers without #include <cstdlib>/<cstring>/<limits>/<mutex>,
    # relying on transitive includes that aren't guaranteed across
    # libstdc++ versions. Force-include them.
    #
    # `sys/cdefs.h` is force-included first so glibc's __GLIBC_PREREQ /
    # __GLIBC_USE macros are defined before any libstdc++ header that
    # gates on them. We use sys/cdefs.h instead of features.h because
    # this project has its own include/features.h that would shadow the
    # glibc one.
    #
    # SHELL: prefix prevents CMake from de-duplicating the repeated
    # -include flag (it would otherwise collapse all filenames into a
    # single -include and drop all but the first).
    # Absolute path for features.h because this project has its own
    # include/features.h that would otherwise shadow glibc's.
    find_path(GLIBC_FEATURES_H features.h
        PATHS /usr/include
        NO_DEFAULT_PATH
        REQUIRED
    )
    # PUBLIC so consumers (the main exe + libtorch-using tests) inherit
    # the same workaround when compiling their own TUs against
    # poker_ppo_lib's headers (which transitively pull libtorch).
    target_compile_options(poker_ppo_lib PUBLIC
        "SHELL:-include ${GLIBC_FEATURES_H}/features.h"
        "SHELL:-include cstring"
        "SHELL:-include cstdlib"
        "SHELL:-include limits"
        "SHELL:-include mutex"
    )
endif()

# ── Example binary ────────────────────────────────────────────────────────────
add_executable(poker_ppo src/main.cpp)
target_link_libraries(poker_ppo PRIVATE poker_ppo_lib)

# ── Tests ─────────────────────────────────────────────────────────────────────
# Opt-in via -DBUILD_TESTS=ON (default ON). Pulls GoogleTest via
# FetchContent inside Game/tests/.
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Game/tests)
endif()
