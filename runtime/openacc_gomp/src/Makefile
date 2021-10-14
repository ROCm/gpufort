include ../rules.mk

.PHONY: all clean

CXX_OBJ = openacc_gomp_wrapper.o
F_OBJ =   openacc_gomp_base.o openacc_gomp.o 

libopenacc_gomp.a: $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a