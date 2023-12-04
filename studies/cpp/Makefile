TEST     ?= overloading namespace-extension
DEPS     ?= 
SOURCES  ?= $(TEST).cpp $(DEPS)
DEPS_OBJ  = $(DEPS:.cpp=.o)

.PHONY: $(TEST) run clean

$(TEST): %: %.cpp $(DEPS_OBJ)
	g++ $^ -o $@

run: test
	./$(TEST)

$(DEPS_OBJ): %.o: %.cpp
	g++ -c $^

clean:
	rm -rf *.o $(TEST)
