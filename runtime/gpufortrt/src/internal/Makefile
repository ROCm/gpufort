include ../../rules.mk

.PHONY: clean

CXX_SRC = ./queue_record_t.cpp \
          ./structured_region_stack_t.cpp \
          ./queue_record_list_t.cpp \
          ./record_list_t.cpp \
          ./record_t.cpp \
          ./auxiliary.cpp \
          ./gpufortrt_core.cpp

CXX_OBJ = $(CXX_SRC:.cpp=.cpp.o) 

all: $(CXX_OBJ)

$(CXX_OBJ): %.cpp.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.mod *.a
