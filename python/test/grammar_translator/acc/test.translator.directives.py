#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import translator.translator as translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

sys.exit()

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
        print(translator.annotatedDoLoop.parseString(snippet)[0])
    except:
        print(" - FAILED",file=sys.stderr)
        print("failed to parse '{}'".format(snippet),file=sys.stderr)
        sys.exit(2)
print(" - PASSED",file=sys.stderr)
