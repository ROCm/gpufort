#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import grammar.grammar as grammar

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

testdata = []
testdata.append("""
!$acc parallel 
!$acc loop
do i = 1, n
  a(i) = 3;
end do
!$acc end parallel
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
!$acc kernels
!$acc loop 
do i = 1, n
  do j = 1, n
    a(i,j) = 3;
  end do
end do
!$acc end kernels
""")

testdata.append("""
!$acc kernels
!$acc loop gang worker
do i = 1, n
  !$acc loop vector
  do j = 1, n
    a(i,j) = 3;
  end do
end do
!$acc end kernels
""")

testdata.append("""
!$acc parallel
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
!$acc end parallel
""")

for snippet in testdata:
    try:
        grammar.accLoopKernel.parseString(snippet)
    except Exception as e:
        print(" - FAILED",file=sys.stderr)
        print("failed to parse '{}'".format(snippet),file=sys.stderr)
        raise e
        sys.exit(2)
print(" - PASSED",file=sys.stderr)
