TESTS = $(shell find . -name "vector-*" -type d)
TESTS_CLEAN = $(shell find . -name "vector*" -type d | sed "s,\./,clean-,g")

.PHONY: $(TESTS) clean codegen.hip

$(TESTS):
	make -C $@ codegen.hip

$(TESTS_CLEAN): clean-%:
	make -C $(subst clean-,,$@) clean

clean: $(TESTS_CLEAN)

codegen.hip: $(TESTS)
