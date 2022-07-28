include ../rules.mk

.PHONY: clean

CXX_OBJ = gpufortrt_c_impl.o
F_OBJ   = gpufortrt_c_bindings.o gpufortrt_base.o gpufortrt.o

$(LIBGPUFORTRT): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a