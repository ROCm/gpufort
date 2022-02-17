TEST     ?= parallel-loop-1
DEPS     ?= 
SOURCES  ?= $(TEST).f90 $(DEPS)
DEPS_OBJ  = $(DEPS:.f90=.o)

LDFLAGS ?= -v -fopenacc # -foffload=amdgcn-amdhsa="-march=gfx900" -offload=-lgfortran
CFLAGS ?= -v -fopenacc  # -foffload=amdgcn-amdhsa="-march=gfx900" -offload=-lgfortran

.PHONY: $(TEST) run clean

$(TEST): %: %.f90 $(DEPS_OBJ)
	gfortran $(LDFLAGS) -cpp $^ -o $@

run: test
	./$(TEST)

$(DEPS_OBJ): %.o: %.f90
	gfortran $(CFLAGS) -cpp -c $^

clean:
	rm -rf *.o *.mod $(TEST)
