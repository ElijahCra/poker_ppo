cmake_minimum_required(VERSION 3.18)
project(poker_ppo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(Game)

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
find_package(Torch REQUIRED)

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

add_executable(poker_ppo src/main.cpp)
target_link_libraries(poker_ppo PRIVATE poker_ppo_lib)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Game/tests)
endif()
