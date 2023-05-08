FC = gfortran

FCFLAGS = -std=f2008 -ffree-line-length-none -cpp
FCFLAGS += -g -ggdb -O0 -fbacktrace -fmax-errors=5

CXX = g++
CXX += -g -ggdb -O0