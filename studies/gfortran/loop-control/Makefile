TEST_F   = loop-control-1 loop-control-2
TEST_CXX = loop-control-cpp

$(TEST_F): %: %.f90
	gfortran -o $@ $^

$(TEST_CXX): %: %.cpp
	g++ -o $@ $^

clean:
	rm -f $(TEST_F) $(TEST_CXX) *.mod
