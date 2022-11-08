HIP_PLATFORM ?= amd
LIBGPUFORTRT  = libgpufortrt_$(HIP_PLATFORM).a

FC      = gfortran -fmax-errors=5
FCFLAGS ?= -std=f2018 -ffree-line-length-none -cpp

#FCFLAGS += -g -ggdb -O0 -fbacktrace -fmax-errors=5 # -DDEBUG=3

CXX       = hipcc
CXXFLAGS ?= -fPIC
#CXXFLAGS += -DBLOCKING_COPIES
#CXXFLAGS += -DEXIT_REGION_SYNCHRONIZE_STREAMS
