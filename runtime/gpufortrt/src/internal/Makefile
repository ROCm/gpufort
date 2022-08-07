include ../rules.mk

.PHONY: clean

CXX_OBJ = ... 
F_OBJ   = ... 

$(LIBGPUFORTRT): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.cpp.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a
