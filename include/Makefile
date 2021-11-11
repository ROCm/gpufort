.PHONY: all headers modules clean_all

SUFFIX  = $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)

all: headers modules

headers:
	gpufort --create-gpufort-headers

modules:
	gpufort -c cudafor.f90

clean_all:
	for expr in "*.json" "*.mod" "*.gpufort_mod" "*-fort2hip.*" "*-gpufort.*" "*.h"; do find . -name "$$expr" -delete; done
