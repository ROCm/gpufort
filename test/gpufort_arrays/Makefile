HIP_TEST_SRC = test_compile_headers.hip.cpp test_gpufort_arrays.hip.cpp
HIP_TEST_APP = $(HIP_TEST_SRC:.hip.cpp=.x)

FORT_TEST_SRC  = test_gpufort_arrays_interop.f03
FORT_TEST_APP  = $(FORT_TEST_SRC:.f03=.x)
FORT_TEST_DEPS = test_gpufort_arrays_interop.hip.o

HIPCC_CFLAGS ?= $(shell gpufort --cpp_config)
FC_CFLAGS    ?= $(shell gpufort --gfortran_config)

.PHONY: clean $(HIP_TEST_APP)

$(HIP_TEST_APP): %.x: %.hip.cpp
	hipcc $(HIPCC_CFLAGS) $^ -o $@

%.hip.o: %.hip.cpp
	hipcc -c $(HIPCC_CFLAGS) $^ -o $@

$(FORT_TEST_APP): %.x: %.f03 $(FORT_TEST_DEPS)
	hipfc $(FC_CFLAGS) -c $^ -o $@

clean:
	rm -f $(HIP_TEST_APP)
