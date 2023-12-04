HIPCC=hipcc-no-inline.pl make clean test
#HIPCC=hipcc.pl HIPCC_COMPILE_FLAGS_APPEND=" -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true" HIPCC_LINK_FLAGS_APPEND=" -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true" make clean test

