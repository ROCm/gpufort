#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import gpufort.translator

testdata=[]
testdata.append("""!$acc kernels if(2>3) self  num_workers(64) num_gangs(8) async(0) wait(1,2) copy(a,b,c) default(PRESENT) private(a,b) reduction(+:a) detach(aaim,a) collapse(3) bind(a) seq tile(4,4)
        do i= 1, N
          a(i) = 10
          b(i) = 5
        end do
        !$acc end kernels""")
testdata.append("""!$acc parallel num_gangs(8)
        do i= 1, N
          a(i) = 10
          b(i) = 5
        end do
        !$acc end parallel""")
testdata.append("""!$acc wait async(0)""")
testdata.append("""!$acc host_data use_device(WS)
        do i= 1, N
          a(i) = 10
          b(i) = 5
        end do
        !$acc end host_data""")
testdata.append("""!$acc data create(IWKX)
        do i= 1, N
          a(i) = 10
          b(i) = 5
        end do
        !$acc end data""")
testdata.append("""!$acc enter data copyin(ICELNOD)""")
testdata.append("""!$acc exit data copyout(X)""")
testdata.append("""!$acc loop independent reduction(+:a)""")
testdata.append("""!$acc routine(JACOBI) seq""")
testdata.append("""!$acc loop independent 
!$acc& private(DETJ,PNX,PNY,PNZ)
!$acc& private(X1,X2,X3,X4,X5,X6,X7,X8)
!$acc& private(Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8)
!$acc& private(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8)
!$acc& private(NODLOCAL)""")
testdata.append("""!$acc atomic read 
        x=3
        !$acc end atomic""")
testdata.append("""!$acc atomic 
        do i = 1,N
        do j = 1,M
        b(indx(i)) = b(indx(i)) + a(i,j) 
        end do
        end do
        !$acc end atomic""")
testdata.append("""!$acc update """)
testdata.append("""!$acc serial create(IWKX)
        do i= 1, N
          a(i) = 10
          b(i) = 5
        end do
        !$acc end serial""")
testdata.append("""!$acc cache (a,b)""")
testdata.append("""!$acc declare link(a)""")
testdata.append("""!$acc kernels
!$acc loop independent gang vector(16)
          do j = jS, jE
            jsL= indexL(j-1)+1
            jeL= indexL(j)
          enddo
!$acc end kernels""")

for i in range(len(testdata)):
    #print(str(i)+".",accKernels.parseString(testdata[i]))
    #results = accKernels.parseString(testdata[i])
    #print(str(i)+".",accClauses.parseString(testdata[i]))
    #results = accClauses.parseString(testdata[i])
    #print(str(i)+".",accConstruct.parseString(testdata[i]))
    results = translator.accConstruct.parseString(testdata[i])
    print(results)
    results[0].printTokens()
    print(results[0].c_str())