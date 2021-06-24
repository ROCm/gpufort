SOURCES = test.f90 testmod.f90
TESTS    = test

DEPS = testmod.o

$(TESTS): %: %.f90 $(DEPS)
	gfortran -cpp $^ -o $@

$(DEPS): %.o: %.f90
	gfortran -cpp -c $^

clean:
	rm -rf *.o *.mod $(TESTS)
