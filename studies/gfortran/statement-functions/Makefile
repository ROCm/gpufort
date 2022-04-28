TEST     ?= statement-functions
DEPS     ?= 
SOURCES  ?= $(TEST).f90 $(DEPS)
DEPS_OBJ  = $(DEPS:.f90=.o)

.PHONY: $(TEST) run clean

$(TEST): %: %.f90 $(DEPS_OBJ)
	gfortran -cpp $^ -o $@

run: test
	./$(TEST)

$(DEPS_OBJ): %.o: %.f90
	gfortran -cpp -c $^

clean:
	rm -rf *.o *.mod $(TEST)
