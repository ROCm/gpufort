F90_FILES         = $(shell find . -name "*.f90")
GPUFORT_MOD_FILES = $(F90_FILES:.f90=.gpufort_mod)

.PHONY: gpufort_modules $(GPUFORT_MOD_FILES)

gpufort_modules: $(GPUFORT_MOD_FILES)

$(GPUFORT_MOD_FILES): %.gpufort_mod: %.f90
	gpufort -c $^

clean:
	rm -f *.gpufort_mod	
