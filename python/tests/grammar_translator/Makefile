TESTS = declarations
GRAMMAR_TESTS = $(shell find . -name "test.grammar.*.py" -execdir basename {} ';')

.PHONY: $(GRAMMAR_TESTS)

$(GRAMMAR_TESTS): %:
	python3 $@
