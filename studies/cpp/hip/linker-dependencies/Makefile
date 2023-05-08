HIPCC ?= hipcc

%.o: %.cpp
	$(HIPCC) -fPIC -c $^ -o $@

test: device_lib.o
	$(HIPCC) -o test main.cpp $^

clean:
	rm -f *.o test

