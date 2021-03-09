TESTS = $(shell find . -name "vector-*" -type d)
TESTS_CLEAN = $(shell find . -name "vector*" -type d | sed "s,\./,clean-,g")

.PHONY: $(TESTS) clean

$(TESTS):
	@echo "make -C $@"

$(TESTS_CLEAN): clean-%: %
	@echo "make -C $@ clean"

clean: $(TESTS_CLEAN)
