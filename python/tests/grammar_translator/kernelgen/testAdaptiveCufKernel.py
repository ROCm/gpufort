#!/usr/bin/env python3
import addtoplevelpath
import translator.translator as translator
           
testdata = []
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
                  
                   wfcatom_d(ig,ipol,ibnd) = wfcatom_d(ig,ipol,ibnd) &*
                             ( 1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP) ) 
               END DO
               !
            END DO
            !
         END DO""")
#testdata.clear()
testdata.append("""!$cuf kernel do(2)<<<*,*>>>
        DO i = 0, howmany-1
           DO j=nsticks_x*nx1+1, nnr_
               f_d(j+i*nnr_) = (0.0_DP,0.0_DP)
           END DO
         END DO""")
testdata.append("""
!$cuf kernel do(2) <<<*,*>>>
  DO ih = 1, nhnp
     DO jh = 1, nhnp
        IF ( jh >= ih ) THEN
           !ijh = jh + ((ih-1)*(2*nhnp-ih))/2  is this faster? Does it matter?
           ijh=ijtoh_d(ih,jh,np)
           IF ( ih == jh ) THEN
              fac = 1.0_dp
           ELSE
              fac = 2.0_dp
           END IF
           becsum_d(ijh,na,1)= becsum_d(ijh,na,1) + fac * &
                   DBLE( becsum_nc_d(ih,1,jh,1) + becsum_nc_d(ih,2,jh,2) )
           IF (domag) THEN
              becsum_d(ijh,na,2)= becsum_d(ijh,na,2) + fac *  &
                   DBLE( becsum_nc_d(ih,1,jh,2) + becsum_nc_d(ih,2,jh,1) )
              becsum_d(ijh,na,3)= becsum_d(ijh,na,3) + fac * DBLE( (0.d0,-1.d0)* &
                  (becsum_nc_d(ih,1,jh,2) - becsum_nc_d(ih,2,jh,1)) )
              becsum_d(ijh,na,4)= becsum_d(ijh,na,4) + fac * &
                   DBLE( becsum_nc_d(ih,1,jh,1) - becsum_nc_d(ih,2,jh,2) )
           END IF
        END IF
     END DO
  END DO
""")
testdata.append("""
!$cuf kernel do(1)
  DO ih = 1, nhnt
     DO jh = 1, nhnt
        ijh=ijtoh_d(ih,jh,np)
        DO kh = 1, nhnt
           IF ( (nhtol_d(kh,np)==nhtol_d(ih,np)).AND. &
                (ABS(nhtoj_d(kh,np)-nhtoj_d(ih,np))<1.d8).AND. &
                (indv_d(kh,np)==indv_d(ih,np)) ) THEN ! same_lj(kh,ih,np)
              DO lh=1,nhnt
                 IF ( (nhtol_d(lh,np)==nhtol_d(jh,np)).AND. &
                      (ABS(nhtoj_d(lh,np)-nhtoj_d(jh,np))<1.d8).AND. &
                      (indv_d(lh,np)==indv_d(jh,np)) ) THEN   !same_lj(lh,jh,np)) THEN
                    DO is1=1,npol
                       DO is2=1,npol
                          fac=becsum_nc_d(kh,is1,lh,is2)
                          becsum_d(ijh,na,1)=becsum_d(ijh,na,1) + DBLE( fac * &
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,1,is2,np) + &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,2,is2,np)  ) )
                          IF (domag) THEN
                            becsum_d(ijh,na,2)=becsum_d(ijh,na,2) + DBLE( fac * &
                                (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) +&
                                 fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ) )
                            becsum_d(ijh,na,3)=becsum_d(ijh,na,3) + DBLE( fac*(0.d0,-1.d0)*&
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) - &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ))
                            becsum_d(ijh,na,4)=becsum_d(ijh,na,4) + DBLE(fac * &
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,1,is2,np) - &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,2,is2,np)  ) )
                        END IF
                     END DO
                  END DO
               END IF
            END DO
         END IF
      END DO
   END DO
END DO
""")
testdata.clear()
testdata.append("""
!$cuf kernel do(2) <<<*,*>>>
  DO ih = 1, nhnp
     DO jh = 1, nhnp
        IF ( jh >= ih ) THEN
           !ijh = jh + ((ih-1)*(2*nhnp-ih))/2  is this faster? Does it matter?
           ijh=ijtoh_d(ih,jh,np)
           IF ( ih == jh ) THEN
              fac = 1.0_dp
           ELSE
              fac = 2.0_dp
           END IF
           becsum_d(ijh,na,1)= becsum_d(ijh,na,1) + fac * &
                   DBLE( becsum_nc_d(ih,1,jh,1) + becsum_nc_d(ih,2,jh,2) )
           IF (domag) THEN
              becsum_d(ijh,na,2)= becsum_d(ijh,na,2) + fac *  &
                   DBLE( becsum_nc_d(ih,1,jh,2) + becsum_nc_d(ih,2,jh,1) )
              !becsum_d(ijh,na,3)= becsum_d(ijh,na,3) + fac * DBLE( (0.d0,-1.d0)* &
              !    (becsum_nc_d(ih,1,jh,2) - becsum_nc_d(ih,2,jh,1)) )
              becsum_d(ijh,na,4)= becsum_d(ijh,na,4) + fac * &
                   DBLE( becsum_nc_d(ih,1,jh,1) - becsum_nc_d(ih,2,jh,2) )
           END IF
        END IF
     END DO
  END DO
""")
testdata.clear()
testdata.append("""
!$cuf kernel do(1)
  DO ih = 1, nhnt
     DO jh = 1, nhnt
        ijh=ijtoh_d(ih,jh,np)
        DO kh = 1, nhnt
           IF ( (nhtol_d(kh,np)==nhtol_d(ih,np)).AND. &
                (ABS(nhtoj_d(kh,np)-nhtoj_d(ih,np))<1.d8).AND. &
                (indv_d(kh,np)==indv_d(ih,np)) ) THEN ! same_lj(kh,ih,np)
              DO lh=1,nhnt
                 IF ( (nhtol_d(lh,np)==nhtol_d(jh,np)).AND. &
                      (ABS(nhtoj_d(lh,np)-nhtoj_d(jh,np))<1.d8).AND. &
                      (indv_d(lh,np)==indv_d(jh,np)) ) THEN   !same_lj(lh,jh,np)) THEN
                    DO is1=1,npol
                       DO is2=1,npol
                          fac=becsum_nc_d(kh,is1,lh,is2)
                          becsum_d(ijh,na,1)=becsum_d(ijh,na,1) + DBLE( fac * &
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,1,is2,np) + &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,2,is2,np)  ) )
                          IF (domag) THEN
                            becsum_d(ijh,na,2)=becsum_d(ijh,na,2) + DBLE( fac * &
                                (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) +&
                                 fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ) )
                            becsum_d(ijh,na,3)=becsum_d(ijh,na,3) + DBLE( fac*(0.d0,-1.d0)*&
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) - &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ))
                            becsum_d(ijh,na,4)=becsum_d(ijh,na,4) + DBLE(fac * &
                               (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,1,is2,np) - &
                                fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,2,is2,np)  ) )
                          END IF
                       END DO
                    END DO
                 END IF
              END DO
           END IF
        END DO
     END DO
  END DO
""")

failedToParse = []
successfullyParsed = []

success = True
for v in testdata: 
    try:
       #translator.cufLoopKernel.parseString(v)
       cSnippet, problemSize, LoopKernelLaunchInfo, identifierNames, localLvalues, loopVars = translator.convertCufLoopKernel2Hip(v,10)
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
