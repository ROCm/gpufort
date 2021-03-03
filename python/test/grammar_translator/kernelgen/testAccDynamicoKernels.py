#!/usr/bin/env python3
import addtoplevelpath
import grammar as grammar
import translator.translator as translator

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
#print(result.cStr())
#print(result.problemSize())

print("k1:")
cSnippet, problemSize, kernelLaunchInfo, identifierNames, localLValues, loopVars, reduction =\
        translator.parseLoopKernel(k1)
print(cSnippet)
print(problemSize[0])

print("k2:")
cSnippet, problemSize, kernelLaunchInfo, identifierNames, localLValues, loopVars, reduction =\
        translator.parseLoopKernel(k2)
print(cSnippet)
print(problemSize[0])

#for i in range(len(k1data)):
#    #print(str(i)+".",accKernels.parseString(k1data[i]))
#    #results = accKernels.parseString(k1data[i])
#    #print(str(i)+".",accClauses.parseString(k1data[i]))
#    #results = accClauses.parseString(k1data[i])
#    #print(str(i)+".",accConstruct.parseString(k1data[i]))
#    results = translator.accConstruct.parseString(k1data[i])
#    print(results)
#    results[0].printTokens()
#    print(results[0].cStr())
