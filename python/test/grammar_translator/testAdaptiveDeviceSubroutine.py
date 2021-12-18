#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import translator.translator as translator
            
testdata = []
testdata.append("""
attributes(global) subroutine test_gpu(A_d,N)
    implicit none
    !
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: N
    real(DP), intent(inout) :: A_d(N)

    i = 1 + threadIdx%x + blockIdx%x * blockDim%x
    if ( i <= 5 ) then
       A_d(i) = A_d(i) + 1
    endif
end subroutine test_gpu
""")
testdata.append("""
attributes(global) subroutine ylmr2_gpu_kernel (lmax,lmax2, ng, g, gg, ylm)
  implicit none
  !
  INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
  REAL(DP), PARAMETER :: pi     = 3.14159265358979323846_DP
  REAL(DP), PARAMETER :: fpi    = 4.0_DP * pi
  integer, intent(in) :: lmax2, ng
  real(DP), intent(in) :: g (3, ng), gg (ng)
  real(DP), intent(out) :: ylm (ng,lmax2)
  !
  ! local variables
  !
  real(DP), parameter :: eps = 1.0d-9
  !real(DP) ::  Q(0:4,0:4)  !Allocate Q for the maximum supported size 

  real(DP) :: cost , sent, phi
  real(DP) :: c, gmod
  integer :: lmax, ig, l, m, lm

  attributes(value)::lmax,lmax2,ng
  attributes(device):: g,gg,Q,ylm
  if (ig <= ng) then
  if (g(1,ig) > eps) then
    phi  = atan( g(2,ig)/g(1,ig) )
  else if (g(1,ig) < -eps) then
    phi  = atan( g(2,ig)/g(1,ig) ) + pi
  else
    phi  = sign( pi/2.d0,g(2,ig) )
  end if
  end if
  return
end subroutine ylmr2_gpu_kernel
""")
testdata.append("""
attributes(global) subroutine ylmr2_gpu_kernel (lmax,lmax2, ng, g, gg, ylm)
  implicit none
  !
  INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
  REAL(DP), PARAMETER :: pi     = 3.14159265358979323846_DP
  REAL(DP), PARAMETER :: fpi    = 4.0_DP * pi
  integer, intent(in) :: lmax2, ng
  real(DP), intent(in) :: g (3, ng), gg (ng)
  real(DP), intent(out) :: ylm (ng,lmax2)
  !
  ! local variables
  !
  real(DP), parameter :: eps = 1.0d-9
  real(DP) ::  Q(0:4,0:4)  !Allocate Q for the maximum supported size 

  real(DP) :: cost , sent, phi
  real(DP) :: c, gmod
  integer :: lmax, ig, l, m, lm

  attributes(value)::lmax,lmax2,ng
  attributes(device):: g,gg,Q,ylm

  ig= threadIdx%x+BlockDim%x*(BlockIdx%x-1)

  if (ig <= ng) then
    !
    if (lmax == 0) then
      ylm(ig,1) =  sqrt (1.d0 / fpi)
      return
    end if
    !
    !  theta and phi are polar angles, cost = cos(theta)
    !
    gmod = sqrt (gg (ig) )
    if (gmod < eps) then
      cost = 0.d0
    else
      cost = g(3,ig)/gmod
    endif
    !
    !  beware the arc tan, it is defined modulo pi
    !
    if (g(1,ig) > eps) then
      phi  = atan( g(2,ig)/g(1,ig) )
    else if (g(1,ig) < -eps) then
      phi  = atan( g(2,ig)/g(1,ig) ) + pi
    else
      phi  = sign( pi/2.d0,g(2,ig) )
    end if
    sent = sqrt(max(0d0,1.d0-cost*cost))
    !
    !  Q(:,l,m) are defined as sqrt ((l-m)!/(l+m)!) * P(:,l,m) where
    !  P(:,l,m) are the Legendre Polynomials (0 <= m <= l)
    !
    Q(0,0) = 1.d0
    Q(1,0) = cost
    Q(1,1) =-sent/sqrt(2.d0)
    c = sqrt (3.d0 / fpi)
    ylm(ig, 1) = sqrt (1.d0 / fpi)* Q(0,0)
    ylm(ig, 2) = c* Q(1,0)
    ylm(ig, 3) = c*sqrt (2.d0)* Q(1,1) * cos (phi)
    ylm(ig, 4) = c*sqrt (2.d0)* Q(1,1) * sin (phi)
  end if
  return
  end subroutine ylmr2_gpu_kernel
end module ylmr2_gpum
""")
testdata.append("""
attributes(global) subroutine ylmr2_gpu_kernel (lmax,lmax2, ng, g, gg, ylm)
  implicit none
  !
  INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
  REAL(DP), PARAMETER :: pi     = 3.14159265358979323846_DP
  REAL(DP), PARAMETER :: fpi    = 4.0_DP * pi
  integer, intent(in) :: lmax2, ng
  real(DP), intent(in) :: g (3, ng), gg (ng)
  real(DP), intent(out) :: ylm (ng,lmax2)
  !
  ! local variables
  !
  real(DP), parameter :: eps = 1.0d-9
  real(DP) ::  Q(0:4,0:4)  !Allocate Q for the maximum supported size 

  real(DP) :: cost , sent, phi
  real(DP) :: c, gmod
  integer :: lmax, ig, l, m, lm

  attributes(value)::lmax,lmax2,ng
  attributes(device):: g,gg,Q,ylm

  ig= threadIdx%x+BlockDim%x*(BlockIdx%x-1)

  if (ig <= ng) then
    !
    if (lmax == 0) then
      ylm(ig,1) =  sqrt (1.d0 / fpi)
      return
    end if
    !
    !  theta and phi are polar angles, cost = cos(theta)
    !
    gmod = sqrt (gg (ig) )
    if (gmod < eps) then
      cost = 0.d0
    else
      cost = g(3,ig)/gmod
    endif
    !
    !  beware the arc tan, it is defined modulo pi
    !
    if (g(1,ig) > eps) then
      phi  = atan( g(2,ig)/g(1,ig) )
    else if (g(1,ig) < -eps) then
      phi  = atan( g(2,ig)/g(1,ig) ) + pi
    else
      phi  = sign( pi/2.d0,g(2,ig) )
    end if
    sent = sqrt(max(0d0,1.d0-cost*cost))
    !
    !  Q(:,l,m) are defined as sqrt ((l-m)!/(l+m)!) * P(:,l,m) where
    !  P(:,l,m) are the Legendre Polynomials (0 <= m <= l)
    !
    Q(0,0) = 1.d0
    Q(1,0) = cost
    Q(1,1) =-sent/sqrt(2.d0)
    c = sqrt (3.d0 / fpi)
    ylm(ig, 1) = sqrt (1.d0 / fpi)* Q(0,0)
    ylm(ig, 2) = c* Q(1,0)
    ylm(ig, 3) = c*sqrt (2.d0)* Q(1,1) * cos (phi)
    ylm(ig, 4) = c*sqrt (2.d0)* Q(1,1) * sin (phi)
    lm = 4
    do l = 2, lmax
      c = sqrt (DBLE(2*l+1) / fpi)
      !
      !  recursion on l for Q(:,l,m)
      !
      do m = 0, l - 2
         Q(l,m) = cost*(2*l-1)/sqrt(DBLE(l*l-m*m))*Q(l-1,m) &
                     - sqrt(DBLE((l-1)*(l-1)-m*m))/sqrt(DBLE(l*l-m*m))*Q(l-2,m)
      end do
      Q(l,l-1) = cost * sqrt(DBLE(2*l-1)) * Q(l-1,l-1)
      Q(l,l)   = - sqrt(DBLE(2*l-1))/sqrt(DBLE(2*l))*sent*Q(l-1,l-1)
      !
      !
      ! Y_lm, m = 0
      !
      lm = lm + 1
      ylm(ig, lm) = c * Q(l,0)
      !
      do m = 1, l
       !
       ! Y_lm, m > 0
       !
         ylm(ig, lm+2*m-1) = c * sqrt(2.d0) * Q(l,m) * cos (m*phi)
       !
       ! Y_lm, m < 0
       !
         ylm(ig, lm+2*m  ) = c * sqrt(2.d0) * Q(l,m) * sin (m*phi)
      end do
      lm=lm+2*l
    end do
  end if
  return
end subroutine ylmr2_gpu_kernel
""")
testdata.append("""
attributes(global) subroutine qvan2_kernel(ngy, ih, jh, np, qmod_d, qg_d, ylmk0_d, lmaxq, nbetam, nlx, dq)
   !-----------------------------------------------------------------------
   !
   !    This routine computes the fourier transform of the Q functions
   !    The interpolation table for the radial fourier trasform is stored 
   !    in qrad.
   !
   !    The formula implemented here is
   !
   !     q(g,i,j) = sum_lm (-i)^l ap(lm,i,j) yr_lm(g^) qrad(g,l,i,j)
   !
   !
   USE kinds, ONLY: DP
   USE us_gpum, ONLY: qrad_d
   USE uspp,      ONLY: lpl_d, lpx_d, ap_d
   USE uspp_gpum, ONLY : indv_d, nhtolm_d
   implicit none
   !
   ! Input variables
   !
   REAL(DP),intent(IN) :: dq
   integer, intent(IN) :: ngy, ih, jh, np, lmaxq, nbetam, nlx
   attributes(value):: ngy, ih, jh, np, lmaxq, nbetam, nlx, dq
   ! ngy   :   number of G vectors to compute
   ! ih, jh:   first and second index of Q
   ! np    :   index of pseudopotentials
   !
   real(DP),intent(IN) :: ylmk0_d (ngy, lmaxq * lmaxq), qmod_d (ngy)
   ! ylmk0 :  spherical harmonics
   ! qmod  :  moduli of the q+g vectors
   !
   ! output: the fourier transform of interest
   !
   real(DP),intent(OUT) :: qg_d (2,ngy)
   attributes(device):: ylmk0_d, qmod_d, qg_d
   !
   !     here the local variables
   !
   real (DP) :: sig
   ! the nonzero real or imaginary part of (-i)^L 
   real (DP), parameter :: sixth = 1.d0 / 6.d0
   !
   integer :: nb, mb, ijv, ivl, jvl, ig, lp, l, lm, i0, i1, i2, i3, ind
   real(DP) :: dqi, qm, px, ux, vx, wx, uvx, pwx, work, qm1
   real(DP) :: uvwx, pwvx, pwux, pxuvx
   !
   ig= threadIdx%x+BlockDim%x*(BlockIdx%x-1)
   !
   return
end subroutine qvan2_kernel
""")

testdata.append("""
attributes(global) subroutine qvan2_kernel(ngy, ih, jh, np, qmod_d, qg_d, ylmk0_d, lmaxq, nbetam, nlx, dq)
   !-----------------------------------------------------------------------
   !
   !    This routine computes the fourier transform of the Q functions
   !    The interpolation table for the radial fourier trasform is stored 
   !    in qrad.
   !
   !    The formula implemented here is
   !
   !     q(g,i,j) = sum_lm (-i)^l ap(lm,i,j) yr_lm(g^) qrad(g,l,i,j)
   !
   !
   USE kinds, ONLY: DP
   USE us_gpum, ONLY: qrad_d
   USE uspp,      ONLY: lpl_d, lpx_d, ap_d
   USE uspp_gpum, ONLY : indv_d, nhtolm_d
   implicit none
   !
   ! Input variables
   !
   REAL(DP),intent(IN) :: dq
   integer, intent(IN) :: ngy, ih, jh, np, lmaxq, nbetam, nlx
   attributes(value):: ngy, ih, jh, np, lmaxq, nbetam, nlx, dq
   ! ngy   :   number of G vectors to compute
   ! ih, jh:   first and second index of Q
   ! np    :   index of pseudopotentials
   !
   real(DP),intent(IN) :: ylmk0_d (ngy, lmaxq * lmaxq), qmod_d (ngy)
   ! ylmk0 :  spherical harmonics
   ! qmod  :  moduli of the q+g vectors
   !
   ! output: the fourier transform of interest
   !
   real(DP),intent(OUT) :: qg_d (2,ngy)
   attributes(device):: ylmk0_d, qmod_d, qg_d
   !
   !     here the local variables
   !
   real (DP) :: sig
   ! the nonzero real or imaginary part of (-i)^L 
   real (DP), parameter :: sixth = 1.d0 / 6.d0
   !
   integer :: nb, mb, ijv, ivl, jvl, ig, lp, l, lm, i0, i1, i2, i3, ind
   real(DP) :: dqi, qm, px, ux, vx, wx, uvx, pwx, work, qm1
   real(DP) :: uvwx, pwvx, pwux, pxuvx
   !
   ig= threadIdx%x+BlockDim%x*(BlockIdx%x-1)
   !
   if (ig <= ngy) then
      !     compute the indices which correspond to ih,jh
      dqi = 1.0_DP / dq
      nb = indv_d (ih, np)
      mb = indv_d (jh, np)
      if (nb.ge.mb) then
         ijv = nb * (nb - 1) / 2 + mb
      else
         ijv = mb * (mb - 1) / 2 + nb
      endif
      ivl = nhtolm_d(ih, np)
      jvl = nhtolm_d(jh, np)
      
      qg_d(1,ig) = 0.d0
      qg_d(2,ig) = 0.d0
      
      qm = qmod_d (ig) * dqi
      px = qm - int (qm)
      ux = 1.d0 - px
      vx = 2.d0 - px
      wx = 3.d0 - px
      i0 = INT( qm ) + 1
      i1 = i0 + 1
      i2 = i0 + 2
      i3 = i0 + 3
      uvx = ux * vx * sixth
      pwx = px * wx * 0.5d0
      
      do lm = 1, lpx_d (ivl, jvl)
         lp = lpl_d (ivl, jvl, lm)
          if (lp == 1) then
             l = 1
             sig = 1.0d0
             ind = 1
          elseif ( lp <= 4) then
             l = 2
             sig =-1.0d0
             ind = 2
          elseif ( lp <= 9 ) then
             l = 3
             sig =-1.0d0
             ind = 1
          elseif ( lp <= 16 ) then
             l = 4
              sig = 1.0d0
             ind = 2
          elseif ( lp <= 25 ) then
             l = 5
             sig = 1.0d0
             ind = 1
          elseif ( lp <= 36 ) then
             l = 6
             sig =-1.0d0
             ind = 2
          else
             l = 7
             sig =-1.0d0
             ind = 1
          endif
          sig = sig * ap_d (lp, ivl, jvl)
               work = qrad_d (i0, ijv, l, np) * uvx * wx + &
                      qrad_d (i1, ijv, l, np) * pwx * vx - &
                      qrad_d (i2, ijv, l, np) * pwx * ux + &
                      qrad_d (i3, ijv, l, np) * px * uvx
          qg_d (ind,ig) = qg_d (ind,ig) + sig * ylmk0_d (ig, lp) * work
      end do
   end if
   !
   return
end subroutine qvan2_kernel
""")
testdata.clear()

testdata.append(
"""
  attributes(global) subroutine zhemv_gpu(N, A, lda, x, y)
    use cudafor
    implicit none

    integer, value                                    :: N, lda
    complex(8), dimension(lda, N), device, intent(in) :: A
    complex(8), dimension(N), device, intent(in)      :: x
    !DIR$ IGNORE_TKR y
    real(8), dimension(2*N), device                   :: y 

    real(8), dimension(BX+1, BX), shared              :: Ar_s
    real(8), dimension(BX+1, BX), shared              :: Ai_s
    real(8), dimension(BX), shared                    :: r_s
    real(8), dimension(BX), shared                    :: i_s

    integer                                           :: tx, ty, ii, jj, i, j, k, istat
    real(8)                                           :: rv1, rv2, iv1, iv2, myrsum, myisum
    real(8)                                           :: Ar, Ai, xrl, xil
    complex(8)                                        :: val

    ! ii,jj is index of top left corner of block
    ii = (blockIdx%y-1) * blockDim%x + 1
    !print*, "ii ", ii

    myrsum = 0.0_8
    myisum = 0.0_8

    tx = threadIdx%x
    ty = threadIdx%y

!    if (ii + (blockIdx%x-1)*blockDim%x > N) return
!
!
!    i = ii + tx - 1
!    if (i <= N) then
!      val =  x(i) ! read part of x for lower triangular multiply
!    endif
!    xrl = dble(val)
!    xil = dimag(val)
!
!    ! Loop over columns (skip all lower triangular blocks)
!!    do jj = ii + (blockIdx%x-1)*blockDim%x, N, gridDim%x*blockDim%x
!      j = jj + ty - 1
!
!      ! Load block into shared memory
!      ! CASE 1: Diagonal block
!      if (ii == jj) then
!
!        ! Load full block into shared memory
!        do k = 0,NTILES-1
!          if (i <= N .and. j + k * blockDim%y <= N) then
!            val = A(i, j + k*blockDim%y)
!            Ar_s(tx, ty + k * blockDim%y) = dble(val)
!            Ai_s(tx, ty + k * blockDim%y) = dimag(val)
!          endif
!        end do
!        
!        call syncthreads()
!
!        ! Reflect to populate lower triangular part with true values of A
!        do k = 0,NTILES-1
!          if (tx > ty + k * blockDim%y) then
!            Ar_s(tx, ty + k * blockDim%y) = Ar_s(ty + k * blockDim%y, tx)
!            Ai_s(tx, ty + k * blockDim%y) = -Ai_s(ty + k * blockDim%y, tx)
!          endif
!        end do
!
!        call syncthreads()
!
!        do k = 0,NTILES-1
!          if (i <= N .and. j + k * blockDim%y <= N ) then
!            Ar = Ar_s(tx, ty + k * blockDim%y); Ai = Ai_s(tx, ty + k * blockDim%y)
!            val = x(j + k*blockDim%y)
!            rv1 = dble(val) ; iv1 = dimag(val)
!            myrsum = myrsum + Ar * rv1 - Ai * iv1
!            myisum = myisum + Ar * iv1 + Ai * rv1
!          endif
!        end do
!
!        !call syncthreads()
!
!      ! CASE 2: Upper triangular block
!      else if (ii < jj) then
!        do k = 0,NTILES-1
!          if (j + k * blockDim%y <= N) then
!            val = A(i, j + k * blockDim%y)
!            Ar = dble(val)
!            Ai = dimag(val)
!          endif
!
!          if (i <= N .and. j + k * blockDim%y <= N ) then
!            val = x(j + k*blockDim%y)
!            rv1 = dble(val) ; iv1 = dimag(val)
!            myrsum = myrsum + Ar * rv1 - Ai * iv1
!            myisum = myisum + Ar * iv1 + Ai * rv1
!          endif
!
!          ! Perform product for symmetric lower block here
!          ! Don't need sync threads since thread is accessing own value
!          !call syncthreads()
!          if (i <= N .and. j + k*blockDim%y <= N) then
!            rv1 = Ar * xrl + Ai * xil
!            iv1 = Ar * xil - Ai * xrl
!          else
!            rv1 = 0.0_8
!            iv1 = 0.0_8
!          endif
!
!          !Partial sum within warps using shuffle
!          rv2 = __shfl_down(rv1,1)
!          rv1 = rv1 + rv2
!          rv2 = __shfl_down(rv1,2)
!          rv1 = rv1 + rv2
!          rv2 = __shfl_down(rv1,4)
!          rv1 = rv1 + rv2
!          rv2 = __shfl_down(rv1,8)
!          rv1 = rv1 + rv2
!          rv2 = __shfl_down(rv1,16)
!          rv1 = rv1 + rv2
!
!          if (tx == 1) then
!            r_s(ty + k*blockDim%y) = rv1
!          endif
!
!          !Partial sum within warps using shuffle
!          iv2 = __shfl_down(iv1,1)
!          iv1 = iv1 + iv2
!          iv2 = __shfl_down(iv1,2)
!          iv1 = iv1 + iv2
!          iv2 = __shfl_down(iv1,4)
!          iv1 = iv1 + iv2
!          iv2 = __shfl_down(iv1,8)
!          iv1 = iv1 + iv2
!          iv2 = __shfl_down(iv1,16)
!          iv1 = iv1 + iv2
!
!          if (tx == 1) then
!            i_s(ty + k*blockDim%y) = iv1
!          endif
!        enddo
!
!        call syncthreads()
!
!        if (ty == 1 .and. jj+tx-1 <= N) then
!          istat = atomicadd(y(2*(jj + tx -1)-1), r_s(tx))
!          istat = atomicadd(y(2*(jj + tx -1)), i_s(tx))
!        endif
!        !call syncthreads()
!
!      endif
!
!      call syncthreads()
!
!    end do

    if (i <= N) then
      istat = atomicadd(y(2*i - 1), myrsum)
      istat = atomicadd(y(2*i), myisum)
    endif
    
  end subroutine zhemv_gpu
""")

testdata.clear()

testdata.append(
"""
  attributes(global) subroutine zhemv_gpu(N, A, lda, x, y)
    use cudafor
    implicit none

    integer, value                                    :: N, lda
    complex(8), dimension(lda, N), device, intent(in) :: A
    complex(8), dimension(N), device, intent(in)      :: x
    !DIR$ IGNORE_TKR y
    real(8), dimension(2*N), device                   :: y 

    real(8), dimension(BX+1, BX), shared              :: Ar_s
    real(8), dimension(BX+1, BX), shared              :: Ai_s
    real(8), dimension(BX), shared                    :: r_s
    real(8), dimension(BX), shared                    :: i_s

    integer                                           :: tx, ty, ii, jj, i, j, k, istat
    real(8)                                           :: rv1, rv2, iv1, iv2, myrsum, myisum
    real(8)                                           :: Ar, Ai, xrl, xil
    complex(8)                                        :: val
    
    ! print*, "ii ", ii
    ii = (blockIdx%y-1) * blockDim%x + 1

    if (i <= N) then
     istat = atomicadd(y(2*i - 1), myrsum)
     istat = atomicadd(y(2*i), myisum)
    endif
end subroutine zhemv_gpu""")


failedToParse = []
successfullyParsed = []

success = True
for v in testdata: 
    try:
       name,argnames,cBody = translator.convertDeviceSubroutine(v,20)
       print("{} -> {}".format(v,cBody))
       print(argnames)
       successfullyParsed.append(v)
    except Exception as e:
       print("failed to parse {}".format(v))
       success = False
       failedToParse.append(v)
       raise e

if success:
    print("SUCCESS!")
else:
    print("Summary: Failed to parse {0} of {1} test inputs".format(len(failedToParse),len(failedToParse)+len(successfullyParsed)))
    print("FAILURE!")