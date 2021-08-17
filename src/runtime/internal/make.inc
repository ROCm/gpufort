CXX      ?= g++
CXXFLAGS ?= 
CXXFLAGS := #$(shell hipconfig --cpp_config) $(CXXFLAGS)
