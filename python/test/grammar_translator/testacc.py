#!/usr/bin/env python3
import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from grammar import *
from translator import *
testdata=[]
testdata.append("""!$acc kernels 
        do i= 1, N
          a(i) = 10
        end do
        !$acc end kernels""")
testdata.append("""!$acc kernels async(0)
        do i= 1, N
          a(i) = 10
        end do
        !$acc end kernels""")
testdata.append("""!$acc kernels copyin(a,b)
        do i= 1, N
          do j = 1, M
          a(i,j) = i * j
        end do
        end do
        !$acc end kernels
        !$acc wait""")
#testdata.append("""!$acc kernels async(0) 
#        !$acc loop independent 
#        do i= 1, N
#          a(i) = 10
#        end do
#        !$acc end kernels""")
#testdata.append("""
#        !$acc kernels
#        !$acc loop reduction(+:BNRM20)
#        do i= 1, N
#          BNRM20= BNRM20+B(3*i-2)**2+B(3*i-1)**2+B(3*i)**2
#        enddo
#        !$acc end kernels
#        """)
testdata.append("""
        !$acc data copyin(a)
        do i= 1, N
          a(i) = 10
        end do
        !$acc end data
        """)
testdata.append("""
        !$acc host_data use_device(a)
        !$acc end host_data
        """)
testdata.clear()
testdata.append("""
!$acc kernels async(0)
!$acc loop independent 
        do k= istart+1, istart+inum
               ii   = 3*NOD_EXPORT(k)
        end do
    !$acc end kernels
    !$acc wait(1)
""")
for i in range(len(testdata)):
    """
    results = block.parseString(testdata[i])
    print(results)
    print(str(i),results)
    results[0][0].printTokens()
    print(results[0][0].cStr())
    """
    results = accConstruct.parseString(testdata[i])
    print(str(i+1)+".",results)
    results[0].printTokens()
    print(flattenBody(results))
    #print(results[0].cStr())
