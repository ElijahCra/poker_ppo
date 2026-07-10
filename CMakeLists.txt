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
    src/optim.cpp
    src/ppo.cpp
    src/rollout.cpp
    src/poker_env.cpp
    src/observation_builder.cpp
    src/league.cpp
    src/metrics_logger.cpp
    src/opponent_manager.cpp
    src/best_response.cpp
    src/lbr.cpp
    src/commands.cpp
    src/escher_trainer.cpp
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

# ── ESCHER research binary (self-contained: small games + libtorch nets) ──────
# Validated on Kuhn/Leduc with exact recursive BR; NOT linked to the HUNL
# poker_env. Inherits libtorch + the force-include workaround.
add_executable(escher
    research/escher_cpp/escher.cpp
    research/escher_cpp/escher_main.cpp
)
target_include_directories(escher PRIVATE research/escher_cpp)
target_link_libraries(escher PRIVATE "${TORCH_LIBRARIES}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(escher PRIVATE
        "SHELL:-include ${GLIBC_FEATURES_H}/features.h"
        "SHELL:-include cstring"
        "SHELL:-include cstdlib"
        "SHELL:-include limits"
        "SHELL:-include mutex"
    )
endif()

# Torch-FREE Phase-1 validator: games + exact BR + CFR+ reproduce the Python
# reference numbers (Kuhn 0.004164, Leduc 0.049151, value -0.0836).
add_executable(escher_validate research/escher_cpp/validate_main.cpp)
target_include_directories(escher_validate PRIVATE research/escher_cpp)

# ── ReBeL research binary (PBS subgame solver + value net; Kuhn/Leduc) ───────
# Faithful Brown et al. 2020 on the exact-judge small games — stage 1 of the
# HUNL port. Reuses the validated escher_cpp games + exact BR.
add_executable(rebel
    research/rebel/rebel.cpp
    research/rebel/rebel_main.cpp
)
target_include_directories(rebel PRIVATE research/rebel research/escher_cpp)
target_link_libraries(rebel PRIVATE "${TORCH_LIBRARIES}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(rebel PRIVATE
        "SHELL:-include ${GLIBC_FEATURES_H}/features.h"
        "SHELL:-include cstring"
        "SHELL:-include cstdlib"
        "SHELL:-include limits"
        "SHELL:-include mutex"
    )
endif()

# ── ReBeL HUNL stage-2: range kernels + street subgame solver ────────────────
# Torch is needed only through poker_env's tensor interface; no CUDA paths.
add_executable(rebel_hunl
    research/rebel/hunl_kernels.cpp
    research/rebel/hunl_solver.cpp
    research/rebel/hunl_gpu.cpp
    research/rebel/hunl_fused.cpp
    research/rebel/hunl_value.cpp
    research/rebel/hunl_play.cpp
    research/rebel/hunl_main.cpp
    src/poker_env.cpp
    src/observation_builder.cpp
    src/lbr.cpp
    src/network.cpp
)
target_include_directories(rebel_hunl PRIVATE research/rebel include)
target_link_libraries(rebel_hunl PRIVATE "${TORCH_LIBRARIES}" Game
                      ${CMAKE_DL_LIBS})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(rebel_hunl PRIVATE
        "SHELL:-include ${GLIBC_FEATURES_H}/features.h"
        "SHELL:-include cstring"
        "SHELL:-include cstdlib"
        "SHELL:-include limits"
        "SHELL:-include mutex"
    )
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
# Opt-in via -DBUILD_TESTS=ON (default ON). Pulls GoogleTest via
# FetchContent inside Game/tests/.
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Game/tests)
endif()
