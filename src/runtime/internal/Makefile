include make.inc

CXX_SOURCES = mapping.cpp
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)
INC         = -I../

$(CXX_OBJECTS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^

all: $(CXX_OBJECTS)

clean:
	rm -f $(CXX_OBJECTS)
	
