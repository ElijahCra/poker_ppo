add_subdirectory(Utility)
add_library(Game INTERFACE)

target_include_directories(Game INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Game INTERFACE HandAbstraction)

set_target_properties(Game PROPERTIES LINKER_LANGUAGE CXX)
