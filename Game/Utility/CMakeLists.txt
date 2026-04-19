add_subdirectory(TwoPlusTwoHandEvaluator)
add_subdirectory(HandAbstraction)
add_library(Utility STATIC Utility.cpp)

target_compile_definitions(Utility PUBLIC PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

