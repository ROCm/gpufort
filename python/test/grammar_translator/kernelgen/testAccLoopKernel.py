#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
from gpufort import grammar
import gpufort.translator

test=\
"""
!$acc parallel loop collapse(2) async
    do l=llb,lle
!DIR$ SIMD
      do ij=ij_begin,ij_end
          ue_gradrot_e(ij+u_right,l)=-ue_gradrot_e(ij+u_right,l)*cgradrot
          ue_gradrot_e(ij+u_lup,l)=-ue_gradrot_e(ij+u_lup,l)*cgradrot
          ue_gradrot_e(ij+u_ldown,l)=-ue_gradrot_e(ij+u_ldown,l)*cgradrot
      enddo
    enddo
"""

test=\
"""!$acc parallel loop collapse(2) async
DO l = llb, lle
!DIR$ SIMD
   DO ij = ij_begin, ij_end
      divu_i(ij, l) = 1./Ai(ij)*(ne(ij, right)*ue_gradivu_e(ij + u_right, l)*le(ij + u_right) + &
                                 ne(ij, rup)*ue_gradivu_e(ij + u_rup, l)*le(ij + u_rup) + &
                                 ne(ij, lup)*ue_gradivu_e(ij + u_lup, l)*le(ij + u_lup) + &
                                 ne(ij, left)*ue_gradivu_e(ij + u_left, l)*le(ij + u_left) + &
                                 ne(ij, ldown)*ue_gradivu_e(ij + u_ldown, l)*le(ij + u_ldown) + &
                                 ne(ij, rdown)*ue_gradivu_e(ij + u_rdown, l)*le(ij + u_rdown))
   ENDDO
ENDDO
"""
test=\
"""
!$acc parallel loop present(ue(:,:), due(:,:), lat_e(:))
DO ij = ij_begin, ij_end
   nn = ij + u_right
   IF (ABS(lat_e(nn)) .gt. rayleigh_limlat) THEN
      !print*, latitude, lat_e(nn)*180./3.14159
      if ( a .gt. 5 ) due(:,:) = due(nn, ll_begin:ll_begin + 1) - (ue(nn, ll_begin:ll_begin + 1)/rayleigh_tau)
      due(nn, ll_begin:ll_begin + 1) = due(nn, ll_begin:ll_begin + 1) - (ue(nn, ll_begin:ll_begin + 1)/rayleigh_tau)
   ENDIF
   nn = ij + u_lup
   IF (ABS(lat_e(nn)) .gt. rayleigh_limlat) THEN
      due(nn, ll_begin:ll_begin + 1) = due(nn, ll_begin:ll_begin + 1) - (ue(nn, ll_begin:ll_begin + 1)/rayleigh_tau)
   ENDIF
   nn = ij + u_ldown
   IF (ABS(lat_e(nn)) .gt. rayleigh_limlat) THEN
      due(nn, ll_begin:ll_begin + 1) = due(nn, ll_begin:ll_begin + 1) - (ue(nn, ll_begin:ll_begin + 1)/rayleigh_tau)
   ENDIF
ENDDO
"""

test=\
"""
!$acc kernels present(ue(:,:), due(:,:), lat_e(:))
due(nn, :) = due(nn, :) - ue(nn, :)/rayleigh_tau
!$acc end kernels
"""

test=\
"""
!$acc parallel loop
DO ij = ij_begin_ext, ij_end_ext
   pk(ij, llm) = ptop + (.5*g)*theta(ij, llm, 1)*rhodz(ij, llm)
END DO
"""

#!$acc parallel loop
#               !DIR$ SIMD
test=\
"""
!$acc parallel loop collapse(2) reduction(max:err) gang private(pk)
               do a=a_omp_begin_ext,a_omp_end_ext
               !DIR$ SIMD
               do ij=ij_omp_begin_ext,ij_omp_end_ext
                  p_ik = pk(ij,l)
                  temp_ik = treff*exp((theta(ij,l,1) + rd*log(p_ik/preff))/cpp)
                  pk(ij,l) = temp_ik
                  ! specific volume v = Rd*T/p = dphi/g/rhodz
                  geopot(ij,l+1) = geopot(ij,l) + (g*rd)*rhodz(ij,l)*temp_ik/p_iki
                  err = max(err,geopot(ij,l+1))
               enddo
               enddo
"""

test=\
"""
     !$acc parallel loop collapse(3) async private(A, dq)
     DO k = kk_begin,kk_end
     DO l = ll_begin,ll_end
!DIR$ SIMD
      DO ij=ij_begin_ext,ij_end_ext
!       CALL gradq(ij,l,ij+t_rup,ij+t_lup,ij+z_up,qi,gradtri(ij+z_up,l,1),gradtri(ij+z_up,l,2),gradtri(ij+z_up,l,3),arr(ij+z_up))

        
        A(1,1)=xyz_i(ij+t_rup,1)-xyz_i(ij,1);  A(1,2)=xyz_i(ij+t_rup,2)-xyz_i(ij,2); A(1,3)=xyz_i(ij+t_rup,3)-xyz_i(ij,3) 
        A(2,1)=xyz_i(ij+t_lup,1)-xyz_i(ij,1);  A(2,2)=xyz_i(ij+t_lup,2)-xyz_i(ij,2); A(2,3)=xyz_i(ij+t_lup,3)-xyz_i(ij,3) 
        A(3,1)=xyz_v(ij+z_up,1);               A(3,2)= xyz_v(ij+z_up,2);             A(3,3)=xyz_v(ij+z_up,3)
    
        dq(1) = qi(ij+t_rup,l)-qi(ij,l)
        dq(2) = qi(ij+t_lup,l)-qi(ij,l)
        dq(3) = 0.0


!        CALL determinant(A(1,1),A(2,1),A(3,1),A(1,2),A(2,2),A(3,2),A(1,3),A(2,3),A(3,3),det)

         a11=A(1,1) ; a12=A(2,1) ; a13=A(3,1)
         a21=A(1,2) ; a22=A(2,2) ; a23=A(3,2)
         a31=A(1,3) ; a32=A(2,3) ; a33=A(3,3)

         x1 =  a11 * (a22 * a33 - a23 * a32)
         x2 =  a12 * (a21 * a33 - a23 * a31)
         x3 =  a13 * (a21 * a32 - a22 * a31)
         det =  x1 - x2 + x3
                 
!        CALL determinant(dq(1),dq(2),dq(3),A(1,2),A(2,2),A(3,2),A(1,3),A(2,3),A(3,3),detx)

         a11=dq(1)  ; a12=dq(2)  ; a13=dq(3)
         a21=A(1,2) ; a22=A(2,2) ; a23=A(3,2)
         a31=A(1,3) ; a32=A(2,3) ; a33=A(3,3)

         x1 =  a11 * (a22 * a33 - a23 * a32)
         x2 =  a12 * (a21 * a33 - a23 * a31)
         x3 =  a13 * (a21 * a32 - a22 * a31)
         detx =  x1 - x2 + x3
        
!        CALL determinant(A(1,1),A(2,1),A(3,1),dq(1),dq(2),dq(3),A(1,3),A(2,3),A(3,3),dety)

         a11=A(1,1) ; a12=A(2,1) ; a13=A(3,1)
         a21=dq(1)  ; a22=dq(2)  ; a23=dq(3)
         a31=A(1,3) ; a32=A(2,3) ; a33=A(3,3)

         x1 =  a11 * (a22 * a33 - a23 * a32)
         x2 =  a12 * (a21 * a33 - a23 * a31)
         x3 =  a13 * (a21 * a32 - a22 * a31)
         dety =  x1 - x2 + x3

!        CALL determinant(A(1,1),A(2,1),A(3,1),A(1,2),A(2,2),A(3,2),dq(1),dq(2),dq(3),detz)

         a11=A(1,1) ; a12=A(2,1) ; a13=A(3,1)
         a21=A(1,2) ; a22=A(2,2) ; a23=A(3,2)
         a31=dq(1)  ; a32=dq(2)  ; a33=dq(3)

         x1 =  a11 * (a22 * a33 - a23 * a32)
         x2 =  a12 * (a21 * a33 - a23 * a31)
         x3 =  a13 * (a21 * a32 - a22 * a31)
         detz =  (x1 - x2 + x3)**3

        gradtri(ij+z_up,l,1) = detx
        gradtri(ij+z_up,l,2) = dety
        gradtri(ij+z_up,l,3) = detz
        arr(ij+z_up) = det
        
      ENDDO
     ENDDO
     ENDDO
"""
test=\
"""
!$acc parallel loop collapse(2) async
    DO l = ll_begin, ll_end
       !DIR$ SIMD
       DO ij = ij_begin, ij_end
          BERNI(ij) = &
             1/(4*Ai(ij))*(le_de(ij + u_right)*u(ij + u_right, l)**2 + &
                           le_de(ij + u_rup)*u(ij + u_rup, l)**2 + &
                           le_de(ij + u_lup)*u(ij + u_lup, l)**2 + &
                           le_de(ij + u_left)*u(ij + u_left, l)**2 + &
                           le_de(ij + u_ldown)*u(ij + u_ldown, l)**2 + &
                           le_de(ij + u_rdown)*u(ij + u_rdown, l)**2)
    ENDDO
    END DO
"""

#translator.LOOP_COLLAPSE_STRATEGY="grid"
translator.LOOP_COLLAPSE_STRATEGY="collapse"
#test = translator.prepareFortranSnippet(test)
#print(test)
#print(grammar.accLoopKernel.parseString(test))
#result = translator.accLoopKernel.parseString(test)[0]
#print(result.c_str())
#print(result.problem_size())
c_snippet, problem_size, kernel_launch_info, identifier_names, localLValues, loop_vars, reduction =\
        translator.parse_loop_kernel(test)
print(c_snippet)

#for i in range(len(testdata)):
#    #print(str(i)+".",accKernels.parseString(testdata[i]))
#    #results = accKernels.parseString(testdata[i])
#    #print(str(i)+".",accClauses.parseString(testdata[i]))
#    #results = accClauses.parseString(testdata[i])
#    #print(str(i)+".",accConstruct.parseString(testdata[i]))
#    results = translator.accConstruct.parseString(testdata[i])
#    print(results)
#    results[0].printTokens()
#    print(results[0].c_str())