TEST     ?= procedure-nesting procedure-nesting-types
DEPS     ?= 
SOURCES  ?= $(TEST).f90 $(DEPS)
DEPS_OBJ  = $(DEPS:.cpp=.o)
LDFLAGS   = -lstdc++

.PHONY: $(TEST) run clean

$(TEST): %: %.f90 $(DEPS_OBJ)
	gfortran -cpp $^ -o $@ $(LDFLAGS)

run: test
	./$(TEST)

$(DEPS_OBJ): %.o: %.cpp
	g++ -c $^

clean:
	rm -rf *.o *.mod $(TEST)
