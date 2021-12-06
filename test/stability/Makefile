CONV        = gpufort
CONV_FLAGS ?= -w -E hip-gpufort-rt

SOURCES = $(shell find . -name "test*.f90")
TESTS=$(SOURCES:.f90=.f90-gpufort.f08)

.PHONY: clean all

all: $(TESTS)

$(TESTS): %.f90-gpufort.f08 : %.f90
	$(CONV) $(CONV_FLAGS) $^ 

clean:
	rm -rf *.f90-* *.gpufort_mod *-fort2hip.hip.cpp
	
	
