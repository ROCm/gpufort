TEST ?= callbacks
TEST_W_DEPS ?= reference-members nested-types bindcpp
DEPS_OBJ = $(addsuffix .o, $(TEST_W_DEPS))
LDFLAGS  = -lstdc++

.PHONY: $(TEST) run clean

$(TEST): %: %.f90
	gfortran -cpp $^ -o $@ $(LDFLAGS)

$(TEST_W_DEPS): %: %.f90 %.o
	gfortran -cpp $^ -o $@ $(LDFLAGS)

run: test
	./$(TEST)

$(DEPS_OBJ): %.o: %.cpp
	g++ -c $^

clean:
	rm -rf *.o *.mod $(TEST) $(TEST_W_DEPS)
