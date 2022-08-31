#!/usr/bin/env bash
rm -f intrinsic.cc
wget https://raw.githubusercontent.com/gcc-mirror/gcc/master/gcc/fortran/intrinsic.cc
grep "add_sym_" intrinsic.cc | grep -o "\"\w\+\"" | grep -o "\w\+" | sort -u > gfortran_intrinsics.inp
