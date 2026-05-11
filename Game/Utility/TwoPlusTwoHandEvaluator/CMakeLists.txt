add_executable(generate_table
        generate_table.cpp
        mtrand.cpp
        pokerlib.cpp
)
target_include_directories(generate_table PUBLIC .)

#add_library(TwoPlusTwoHandEvaluator generate_table.cpp)


