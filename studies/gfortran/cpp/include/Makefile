SOURCES = test.f90
TESTS    = $(SOURCES:.f90=)

$(TESTS): %: %.f90
	gfortran -cpp $^ -o $@

clean:
	rm -rf *.o *.mod $(TESTS)
