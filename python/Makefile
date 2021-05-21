TEST_FOLDERS = test/grammar_translator/openacc test/grammar_translator/cudafortran

TEST_COLLECTIONS            := $(shell echo "$(TEST_FOLDERS)" | sed "s,/,.,g")
GRAMMAR_TEST_COLLECTIONS    := $(addsuffix .grammar,$(TEST_COLLECTIONS))
TRANSLATOR_TEST_COLLECTIONS := $(addsuffix .translator,$(TEST_COLLECTIONS))

.PHONY: test test.grammar $(TEST_COLLECTIONS) $(GRAMMAR_TEST_COLLECTIONS) $(TRANSLATOR_TEST_COLLECTIONS)

test: $(TEST_COLLECTIONS)

test.grammar: $(GRAMMAR_TEST_COLLECTIONS)

test.translator: $(TRANSLATOR_TEST_COLLECTIONS)

$(TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.,/,g")

$(GRAMMAR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.grammar\>,,g" | sed "s,\.,/,g") test.grammar

$(TRANSLATOR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.translator\>,,g" | sed "s,\.,/,g") test.translator
