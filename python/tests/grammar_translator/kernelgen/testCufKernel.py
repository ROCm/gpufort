#!/usr/bin/env python3
import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
import translator.translator
            
testdata = ["""
!$cuf kernel do(1) <<<*,*,0,stream>>>
           DO i=1, ldz * nsl
              c_d( i ) = c_d( i ) * tscale
           END DO
        ELSE
""",
"""
!$cuf kernel do(1) <<<*,*,0,stream>>>
           DO i=1, ldz * nsl
              cout_d( i ) = c_d( i ) * tscale
           END DO
""",
"""
!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
        DO k=1, nzl
           DO j=1, ldy
             DO i=1, ldx
                r_d(i,j,k) = r_d(i,j,k) * tscale
              END DO
           END DO
        END DO
""",
"""
!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
        DO k=1, nzl
           DO j=1, ldy
             DO i=1, ldx
                r_d(i,j,k) = temp_d(j,k,i) * tscale
              END DO
           END DO
        END DO
""",
"""
!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
        DO k=1, nzl
           DO i=1, ldx
              DO j=1, ldy
                temp_d(j,k,i) = r_d(i,j,k)
              END DO
           END DO
        END DO
""",
"""
!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
        DO k=1, nzl
           DO j=1, ldy
             DO i=1, ldx
                r_d(i,j,k) = temp_d(j,k,i)
              END DO
           END DO
        END DO
""",
"""
!$cuf kernel do(1) <<<*,(16,16,1),0,stream>>>
        DO i=1, ldx*ldy*ldz*howmany
           f_d( i ) = f_d( i ) * tscale
        END DO
""",
"""
        !$cuf kernel do(1) <<<*,*,0,stream>>>
        DO i=1, nx*ny*nz
           f_d( i ) = f_d( i ) * tscale
        END DO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
        DO k = 1, ncp_ (me)
           DO i = 1, npp_ ( gproc )
             f_aux_d( kdest + i + (k-1)*nppx ) = f_in_d( kfrom + i + (k-1)*nr3x )
           END DO
        END DO
""",
"""
     !$cuf kernel do (1) <<<*,*>>>
     do i = lbound(f_aux_d,1), ubound(f_aux_d,1)
       f_aux_d(i) = (0.d0, 0.d0)
     end do
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx
                 mc = p_ismap_d( cuf_i + ioff )
                 f_aux_d( mc + ( cuf_j - 1 ) * nnp ) = f_in_d( cuf_j + it )
              ENDDO
           ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 !
                 mc = p_ismap_d( cuf_i + ioff )
                 !
                 it = (cuf_i-1) * nppx + ( gproc - 1 ) * sendsiz
                 !
                 f_aux_d( mc + ( cuf_j - 1 ) * nnp ) = f_in_d( cuf_j + it )
                 !
              ENDDO
            ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 mc = p_ismap_d( cuf_i + ioff )
                 it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx
                 f_in_d( cuf_j + it ) = f_aux_d( mc + ( cuf_j - 1 ) * nnp )
              ENDDO
           ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 !
                 mc = p_ismap_d( cuf_i + ioff )
                 !
                 it = (cuf_i-1) * nppx + ( gproc - 1 ) * sendsiz
                 !
                 f_in_d( cuf_j + it ) = f_aux_d( mc + ( cuf_j - 1 ) * nnp )
              ENDDO
              !
           ENDDO
           !
""",
"""
!$cuf kernel do(2) <<<*,*>>>
        DO k = 1, ncp_ (me)
           DO i = 1, npp_ ( gproc )
             f_in_d( kfrom + i + (k-1)*nr3x ) = f_aux_d( kdest + i + (k-1)*nppx )
           END DO
        END DO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
        DO k = 1, batchsize * ncpx
           DO i = 1, npp_ ( gproc )
             f_aux_d( kdest + i + (k-1)*nppx ) = f_in_d( kfrom + i + (k-1)*nr3x )
           END DO
        END DO
""",
"""
     !$cuf kernel do (1) <<<*,*>>>
     DO i = lbound(f_aux_d,1), ubound(f_aux_d,1)
       f_aux_d(i) = (0.d0, 0.d0)
     END DO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx
                 mc = p_ismap_d( cuf_i + ioff )
                 f_aux_d( mc + ( cuf_j - 1 ) * nnp ) = f_in_d( cuf_j + it )
              ENDDO
           ENDDO
""",
"""
!$cuf kernel do(3) <<<*,*>>>
           DO i = 0, batchsize-1
              DO cuf_j = 1, npp
                 DO cuf_i = 1, nswip
                    !
                    mc = p_ismap_d( cuf_i + ioff )
                    !
                    it = (cuf_i-1) * nppx + ( gproc - 1 ) * sendsiz + i*nppx*ncpx
                    !
                    f_aux_d( mc + ( cuf_j - 1 ) * nnp + i*nnr ) = f_in_d( cuf_j + it )
                 ENDDO
               ENDDO
             ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
           DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 mc = p_ismap_d( cuf_i + ioff )
                 it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx
                    f_in_d( cuf_j + it ) = f_aux_d( mc + ( cuf_j - 1 ) * nnp )
              ENDDO
           ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
        DO k = 1, batchsize * ncpx
           DO i = 1, npp_ ( gproc )
             f_in_d( kfrom + i + (k-1)*nr3x ) = f_aux_d( kdest + i + (k-1)*nppx )
           END DO
        END DO
""",
"""
   !$cuf kernel do (1) <<<*,*,0,dfft%a2a_comp>>>
   do i = lbound(f_aux_d,1), ubound(f_aux_d,1)
     f_aux_d(i) = (0.d0, 0.d0)
   end do
""",
"""
!$cuf kernel do(3) <<<*,*,0,dfft%a2a_comp>>>
         DO i = 0, batchsize-1
            DO cuf_j = 1, npp
               DO cuf_i = 1, nswip
                  it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx + i*nppx*ncpx
                  mc = p_ismap_d( cuf_i + ioff )
                  f_aux_d( mc + ( cuf_j - 1 ) * nnp + i*nnr ) = f_aux2_d( cuf_j + it )
               ENDDO
            ENDDO
         ENDDO
""",
"""
!$cuf kernel do(3) <<<*,*,0,dfft%a2a_comp>>>
         DO i = 0, batchsize-1
            DO cuf_j = 1, npp
              DO cuf_i = 1, nswip
                 !
                 mc = p_ismap_d( cuf_i + ioff )
                 !
                 it = (cuf_i-1) * nppx + ( gproc - 1 ) * sendsiz + i*nppx*ncpx
                 !
                 f_aux_d( mc + ( cuf_j - 1 ) * nnp + i*nnr ) = f_aux2_d( cuf_j + it )
               ENDDO
            ENDDO
         ENDDO
""",
"""
!$cuf kernel do(3) <<<*,*,0,dfft%a2a_comp>>>
         DO i = 0, batchsize-1
            DO cuf_j = 1, npp
               DO cuf_i = 1, nswip
                  mc = p_ismap_d( cuf_i + ioff )
                  it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx + i*nppx*ncpx
                  f_aux2_d( cuf_j + it ) = f_aux_d( mc + ( cuf_j - 1 ) * nnp + i*nnr )
               ENDDO
            ENDDO
         ENDDO
""",
"""
!$cuf kernel do(3) <<<*,*, 0, dfft%a2a_comp>>>
         DO i = 0, batchsize-1
            DO cuf_j = 1, npp
               DO cuf_i = 1, nswip
                 !
                 mc = p_ismap_d( cuf_i + ioff )
                 !
                 it = (cuf_i-1) * nppx + ( gproc - 1 ) * sendsiz + i*nppx*ncpx
                 !
                 f_aux2_d( cuf_j + it ) = f_aux_d( mc + ( cuf_j - 1 ) * nnp + i*nnr )
               ENDDO
            ENDDO
         ENDDO
""",
"""
        !$cuf kernel do (2) <<<*,*,0,stream>>>
        DO i = 1, aux
           DO j = 1, my_nr2p
              it = ( iproc2 - 1 ) * sendsize + (i-1)*nr2px
              m3 = (i-1)/nr1p__d(iproc2)+1 ; i1  = mod(i-1,nr1p__d(iproc2))+1 ;  m1 = indx_d(i1,iproc2)
              icompact = m1 + (m3-1)*nr1x*my_nr2p + (j-1)*nr1x
           !DO j = 1, my_nr2p
              !f_in( j + it ) = f_aux( m1 + (j-1)*nr1x + (m3-1)*nr1x*my_nr2p )
              f_in_d( j + it ) = f_aux_d( icompact )
           !   icompact = icompact + nr1x
           ENDDO
        ENDDO
""",
"""
     !$cuf kernel do (1) <<<*,*,0,stream>>>
     do i = 1, nxx_
       f_in_d(i) = (0.d0, 0.d0)
     end do
""",
"""
!$cuf kernel do (1) <<<*,*,0,desc%stream_scatter_yz(1)>>>
     DO i = 1, desc%my_nr3p*my_nr1p_*nr2x
       f_aux_d(i) = (0.d0, 0.d0)
     END DO
""",
"""
!$cuf kernel do(2) <<<*,*,0,desc%stream_scatter_yz(iproc3)>>>
           DO i = 1, aux ! was ncp_(iproc3)
              DO k = 1, desc%my_nr3p
                 it = it0 + (i-1)*nr3px
                 mc = desc_ismap_d( i + ioff ) ! this is  m1+(m2-1)*nr1x  of the  current pencil
                 m1 = mod (mc-1,nr1x) + 1 ; m2 = (mc-1)/nr1x + 1
                 i1 = m2 + ( ir1p__d(m1) - 1 ) * nr2x + (k-1)*nr2x*my_nr1p_

                 f_aux_d( i1 ) = f_in_d( k + it )
                 !i1 = i1 + desc%nr2x*my_nr1p_
              ENDDO
            ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*,0,desc%stream_scatter_yz(iproc3)>>>
           DO i = 1, aux
              DO k = 1, desc%my_nr3p
                 it = it0 + (i-1)*nr3px
                 mc = desc_ismap_d( i + ioff ) ! this is  m1+(m2-1)*nr1x  of the  current pencil
                 m1 = mod (mc-1,nr1x) + 1 ; m2 = (mc-1)/nr1x + 1
                 i1 = m2 + ( ir1p__d(m1) - 1 ) * nr2x + (k-1)*(nr2x * my_nr1p_)

                 f_in_d( k + it ) = f_aux_d( i1 )
                 !i1 = i1 + desc%nr2x * my_nr1p_
              ENDDO
           ENDDO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
       DO k = 1, aux
          DO i = nr3, nr3x
             f_in_d( (k-1)*nr3x + i ) = (0.d0, 0.d0)
          END DO
       END DO
""",
"""
     !$cuf kernel do (1) <<<*,*>>>
     do i = 1, nxx_
       f_aux_d(i) = (0.d0, 0.d0)
     end do
""",
"""
!$cuf kernel do(3) <<<*,*>>>
       DO j=0, howmany-1
          DO k = 1, aux
             DO i = nr3, nr3x
                f_in_d( j*ncpx*nr3x + (k-1)*nr3x + i) = 0.0d0
             END DO
          END DO
       END DO
""",
"""
!$cuf kernel do(2) <<<*,*>>>
  do ir3 =1, dffts%my_nr3p
     do i=1, nxyp
        off    = nr1x*my_nr2p*(ir3-1)
        tg_off = nr1x*nr2x   *(ir3-1) + nr1x*my_i0r2p
        tg_v_d(tg_off + i) = v_d(off+i)
     end do
  end do
""",
"""
!$cuf kernel do(2) <<<*,*>>>
  do ir3 =1, dffts%my_nr3p
     do i=1, nxyp
        off    = nr1x*my_nr2p*(ir3-1)
        tg_off = nr1x*nr2x   *(ir3-1) + nr1x*my_i0r2p
        tg_v_d(tg_off + i) = v_d(off+i)
     end do
  end do
""",
"""
!$cuf kernel do(1)
        DO ig=1,desc%ngw
           vout1(ig) = CMPLX( DBLE(vin(nl(ig))+vin(nlm(ig))),AIMAG(vin(nl(ig))-vin(nlm(ig))),kind=DP)
           vout2(ig) = CMPLX(AIMAG(vin(nl(ig))+vin(nlm(ig))),-DBLE(vin(nl(ig))-vin(nlm(ig))),kind=DP)
        END DO
     ELSE
""",
"""
!$cuf kernel do(1)
        DO ig=1,desc%ngw
           vout1(ig) = vin(nl(ig))
        END DO
""",
"""
        !$cuf kernel do(1)<<<*,*,0,stream>>>
        do i=nsticks_x*nx1+1, nnr_
            f_d(i) = (0.0_DP,0.0_DP)
        end do
""",
"""
           !$cuf kernel do(1)<<<*,*,0,stream>>>
           do i=nsticks_z*nx3+1, nnr_
               aux_d(i) = (0.0_DP,0.0_DP)
           end do
""",
"""
           !$cuf kernel do(1)<<<*,*,0,stream>>>
           do i=nsticks_z*nx3+1, nnr_
               f_d(i) = (0.0_DP,0.0_DP)
           end do
""",
"""
!$cuf kernel do(2)<<<*,*>>>
        DO i = 0, howmany-1
           DO j=nsticks_x*nx1+1, nnr_
               f_d(j+i*nnr_) = (0.0_DP,0.0_DP)
           END DO
        END DO
""",
"""
!$cuf kernel do(2)<<<*,*>>>
        DO i = 0, howmany-1
           DO j=nsticks_z*nx3+1, nnr_
                f_d(j+i*nnr_) = (0.0_DP,0.0_DP)
           END DO
        END DO
"""]


testdata.append("""!$cuf kernel do(3)
         DO ibnd = 1, n_starting_atomic_wfc
            !
            DO ipol = 1, npol
               !
               DO ig = 1, ngk_ik
                  !
                  rnd_idx = 2 * ((ig-1) + ( (ipol-1) + (ibnd-1) * npol ) * ngk_ik) + 1
                  rr  = randy_d(rnd_idx)
                  arg = tpi * randy_d(rnd_idx+1)
                  !
                   wfcatom_d(ig,ipol,ibnd) = wfcatom_d(ig,ipol,ibnd) &*
                             ( 1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP) )
               END DO
               !
            END DO
            !
         END DO""")

failedToParse = []
successfullyParsed = []

success = True
for v in testdata: 
    try:
       cSnippet, problemSize, LoopKernelLaunchInfo, identifierNames, localLvalues, loopVars = translator.convertCufLoopKernel(v)
       print("{} -> {}".format(v,cSnippet))
       successfullyParsed.append(v)
       print(localLvalues)
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
