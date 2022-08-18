HIP_PLATFORM ?= amd
LIBGPUFORTRT  = libgpufortrt_$(HIP_PLATFORM).a

FC      = gfortran -fmax-errors=5
FCFLAGS ?= -std=f2008 -ffree-line-length-none -cpp

FCFLAGS += -DNULLPTR_MEANS_NOOP -DDELETE_NORECORD_MEANS_NOOP
#FCFLAGS += -DBLOCKING_COPIES
#FCFLAGS += -DEXIT_REGION_SYNCHRONIZE_STREAMS
#FCFLAGS += -g -ggdb -O0 -fbacktrace -fmax-errors=5 # -DDEBUG=3

CXX       = hipcc