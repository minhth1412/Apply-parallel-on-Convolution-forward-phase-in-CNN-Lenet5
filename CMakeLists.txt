cmake_minimum_required (VERSION 3.0)
project (myproject)

# Include Eigen library from third_party directory
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Find all source files in the current directory
# and save their names to the DIR_SRCS variable
aux_source_directory(. DIR_SRCS)

# Add the src subdirectory
add_subdirectory(src)

# Specify the target to be generated
add_executable(train ${DIR_SRCS})

# Add linking library
target_link_libraries(train MiniDNNLib)
