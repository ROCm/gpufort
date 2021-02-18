TEST_FOLDERS = tests/grammar_translator/acc

TESTS_COLLECTIONS = $(TEST_FOLDERS:/=.)

.PHONY: tests $(TESTS_COLLECTIONS)

$(TESTS_COLLECTIONS): %:
	make -C $@

tests: $(TESTS_COLLECTIONS)
