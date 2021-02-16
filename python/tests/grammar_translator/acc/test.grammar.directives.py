#!/usr/bin/env python3
import addtoplevelpath
import sys
import grammar.grammar as grammar

testdata = []
testdata.append("""
!$acc loop 
do i = 1, n
  a(i) = 3;
end do
""")
testdata.append("""
!$acc kernels loop 
do i = 1, n
  a(i) = 3;
end do
""")

testdata.append("""
!$acc parallel loop 
do i = 1, n
  a(i) = 3;
end do
""")

testdata.append("""
!$acc loop 
do i = 1, n
  do j = 1, n
    a(i,j) = 3;
  end do
end do
""")

testdata.append("""
!$acc loop gang worker
do i = 1, n
  !$acc loop vector
  do j = 1, n
    a(i,j) = 3;
  end do
end do
""")

testdata.append("""
!$acc loop gang
do i = 1, n
!$acc loop worker
  do i = 1, n
    !$acc loop vector
    do j = 1, n
      a(i,j,k) = 3;
    end do
  end do
end do
""")

for snippet in testdata:
    try:
        grammar.annotatedDoLoop.parseString(snippet)
    except:
        print("FAILED to parse '{}'".format(snippet),file=sys.stderr)
print("PASSED",file=sys.stderr)
