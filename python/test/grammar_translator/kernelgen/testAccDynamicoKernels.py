#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
from gpufort import grammar
import gpufort.translator

k1="""
!$acc kernels collapse(2) async
DO l = llm - 1, 1, -1
   !DIR$ SIMD
   DO ij = ij_omp_begin_ext, ij_omp_end_ext
      pk(ij, l) = pk(ij, l + 1) + (.5*g)*(rhodz(ij, l) + rhodz(ij, l + 1))
   END DO
END DO
"""

k2="""
!$acc kernels async default(present) collapse(2)
    DO l = ll_begin, ll_end
       y(:, l) = y(:, l) + w*dy(:, l)
    ENDDO
"""

#translator.LOOP_COLLAPSE_STRATEGY="grid"
translator.LOOP_COLLAPSE_STRATEGY="collapse"

#k1 = translator.prepareFortranSnippet(k1)
#print(k1)
#print(grammar.accLoopKernel.parseString(k1))
#result = translator.accLoopKernel.parseString(k1)[0]
#print(result.c_str())
#print(result.problem_size())

print("k1:")
c_snippet, problem_size, kernel_launch_info, identifier_names, localLValues, loop_vars, reduction =\
        translator.parse_loop_kernel(k1)
print(c_snippet)
print(problem_size[0])

print("k2:")
c_snippet, problem_size, kernel_launch_info, identifier_names, localLValues, loop_vars, reduction =\
        translator.parse_loop_kernel(k2)
print(c_snippet)
print(problem_size[0])

#for i in range(len(k1data)):
#    #print(str(i)+".",accKernels.parseString(k1data[i]))
#    #results = accKernels.parseString(k1data[i])
#    #print(str(i)+".",accClauses.parseString(k1data[i]))
#    #results = accClauses.parseString(k1data[i])
#    #print(str(i)+".",accConstruct.parseString(k1data[i]))
#    results = translator.accConstruct.parseString(k1data[i])
#    print(results)
#    results[0].printTokens()
#    print(results[0].c_str())