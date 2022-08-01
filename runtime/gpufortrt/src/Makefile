include ../rules.mk

.PHONY: clean

CXX_OBJ = gpufortrt_c_impl.cpp.o gpufort_c_bindings.cpp.o 
F_OBJ   = gpufortrt_c_bindings.o gpufortrt_core.o gpufortrt.o

$(LIBGPUFORTRT): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.cpp.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a