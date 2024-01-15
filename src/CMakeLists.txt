# Find all source files in the current directory
# and save their names to the DIR_LIB_SRCS variable
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)
aux_source_directory(./layer/custom DIR_LIB_SRCS)

# Generate a linking library
add_library(MiniDNNLib ${DIR_LIB_SRCS} "layer/custom/gpu_conv_forward.h" "layer/custom/gpu_utils.h")
