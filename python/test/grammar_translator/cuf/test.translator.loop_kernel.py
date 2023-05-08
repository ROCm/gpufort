#!/usr/bin/env python3


import os,sys
import addtoplevelpath
from gpufort import translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

testdata = []
testdata.append("""
!$cuf kernel do(1) 
do i = 1, n
  a(i) = 3;
end do
""")
testdata.append("""
!$cuf kernel do(1) <<<*,*>>>
do i = 1, n
  a(i) = 3;
end do
""")
testdata.append("""
!$cuf kernel do(1) <<<a,b>>>
do i = 1, n
  a(i) = 3;
end do
""")

testdata.append("""
!$cuf kernel do(1) <<<grid, tBlock>>>
do i=1,N
  y(i) = y(i) + a*x(i)
end do
""")

for snippet in testdata:
    try:
        translator.parse_loopnest(snippet.splitlines())
    except Exception as e:
        print(" - FAILED",file=sys.stderr)
        print("failed to parse '{}'".format(snippet),file=sys.stderr)
        raise e
        sys.exit(2)
print(" - PASSED",file=sys.stderr)