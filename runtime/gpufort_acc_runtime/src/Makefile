include ../rules.mk

.PHONY: clean

CXX_OBJ = gpufort_acc_runtime_c_impl.o
F_OBJ   = gpufort_acc_runtime_base.o gpufort_acc_runtime.o

$(LIBGPUFORT_ACC): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a
