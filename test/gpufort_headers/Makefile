TEST_SRC = test_compile_headers.hip.cpp test_gpufort_arrays.hip.cpp
TEST_APP = $(TEST_SRC:.hip.cpp=.x)

.PHONY: generate_headers clean

$(TEST_APP): %.x: %.hip.cpp
	hipcc $^ -o $@
	
generate_headers:
	gpufort --create-gpufort-headers

clean:
	rm $(TEST_APP) -f gpufort.h gpufort_reductions.h gpufort_arrays.h
