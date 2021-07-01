GRAMMAR_TRANSLATOR_TEST_COLLECTIONS_PREFIXES  := test.grammar_translator.openacc test.grammar_translator.cudafortran \
	test.grammar_translator.preprocessor
GRAMMAR_TEST_COLLECTIONS                      := $(addsuffix .grammar,$(GRAMMAR_TRANSLATOR_TEST_COLLECTIONS_PREFIXES))
TRANSLATOR_TEST_COLLECTIONS                   := $(addsuffix .translator,$(GRAMMAR_TRANSLATOR_TEST_COLLECTIONS_PREFIXES))
INDEXER_TEST_COLLECTIONS = test.indexer

TEST_COLLECTIONS = $(GRAMMAR_TEST_COLLECTIONS)\
		    $(TRANSLATOR_TEST_COLLECTIONS)\
		    $(INDEXER_TEST_COLLECTIONS)

.PHONY: test test.grammar test.translator test.indexer\
	 $(GRAMMAR_TEST_COLLECTIONS)\
	 $(TRANSLATOR_TEST_COLLECTIONS)\
	 $(INDEXER_TEST_COLLECTIONS)

test: $(TEST_COLLECTIONS)
	echo $(TEST_COLLECTIONS)

test.grammar: $(GRAMMAR_TEST_COLLECTIONS)

test.translator: $(TRANSLATOR_TEST_COLLECTIONS)

$(GRAMMAR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.grammar\>,,g" | sed "s,\.,/,g") test.grammar

$(TRANSLATOR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.translator\>,,g" | sed "s,\.,/,g") test.translator

test.indexer: $(INDEXER_TEST_COLLECTIONS)
$(INDEXER_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.,/,g") test.indexer clean
