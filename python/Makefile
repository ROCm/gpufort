TRANSLATOR_TEST_COLLECTIONS_PREFIXES  := test.translator.acc test.translator.cuf \
	test.translator.linemapper test.translator
GRAMMAR_TEST_COLLECTIONS                      := $(addsuffix .grammar,$(TRANSLATOR_TEST_COLLECTIONS_PREFIXES))
TRANSLATOR_TEST_COLLECTIONS                   := $(addsuffix .translator,$(TRANSLATOR_TEST_COLLECTIONS_PREFIXES))

UTIL_TEST_COLLECTIONS = test.util

INDEXER_TEST_COLLECTIONS = test.indexer

LINEMAPPER_TEST_COLLECTIONS = test.linemapper

TEST_COLLECTIONS =  $(GRAMMAR_TEST_COLLECTIONS)\
		    $(TRANSLATOR_TEST_COLLECTIONS)\
		    $(UTIL_TEST_COLLECTIONS)\
		    $(INDEXER_TEST_COLLECTIONS)\
		    $(LINEMAPPER_TEST_COLLECTIONS)

.PHONY: test test.grammar test.translator test.indexer test.linemapper\
	 $(GRAMMAR_TEST_COLLECTIONS)\
	 $(TRANSLATOR_TEST_COLLECTIONS)\
	 $(UTIL_TEST_COLLECTIONS)\
	 $(INDEXER_TEST_COLLECTIONS)\
	 $(LINEMAPPER_TEST_COLLECTIONS)

test: $(TEST_COLLECTIONS)
	echo $(TEST_COLLECTIONS)

test.grammar: $(GRAMMAR_TEST_COLLECTIONS)

test.translator: $(TRANSLATOR_TEST_COLLECTIONS)

$(GRAMMAR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.grammar\>,,g" | sed "s,\.,/,g") test.grammar

$(TRANSLATOR_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.translator\>,,g" | sed "s,\.,/,g") test.translator

test.util: $(UTIL_TEST_COLLECTIONS)

$(UTIL_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.,/,g") test.util clean

test.indexer: $(INDEXER_TEST_COLLECTIONS)

$(INDEXER_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.,/,g") test.indexer clean

test.linemapper: $(LINEMAPPER_TEST_COLLECTIONS)

$(LINEMAPPER_TEST_COLLECTIONS): %:
	make -C $(shell echo "$@" | sed "s,\.,/,g") test.linemapper clean