#!/usr/bin/env python3
import addtoplevelpath
import sys
import test
import translator.translator as translator

#testdata = []
#testdata.append("2.d0*f1")
#
#test.run(
#   expression     = translator.arithmeticExpression,
#   testdata       = testdata,
#   tag            = None,
#   raiseException = True
#)

print(translator.assignment.parseString("a(:,:)=b(:,:)+c(:,:)")[0].cStr(unpack=True))
print(translator.assignment.parseString("a(1:3,:)=b(1:3,:)+c(1:3,:)")[0].cStr(unpack=True))
print(translator.assignment.parseString("a(:,1)=b(:,1)+c(:,1)")[0].cStr(unpack=True))

print(translator.power.parseString("""becsum_d(ijh,na,3)**2""")[0].cStr())

print(translator.assignment.parseString("""becsum_d(ijh,na,3)=becsum_d(ijh,na,3)**2""")[0].cStr())
sys.exit()

print(translator.conversions.parseString("""
DBLE( fac )
""")[0].cStr())
print(translator.conversions.parseString("""
DBLE( fac * fcoef_d(kh,ih,is1,1,np) * fcoef_d(jh,lh,2,is2,np) )
""")[0].cStr())
print(translator.conversions.parseString("""
DBLE( fac * (fcoef_d(kh,ih,is1,1,np) * fcoef_d(jh,lh,2,is2,np) ))
""")[0].cStr())
print(translator.conversions.parseString("""
DBLE( fac * (fcoef_d(kh,ih,is1,1,np) * fcoef_d(jh,lh,2,is2,np) ) )
""")[0].cStr())
print(translator.conversions.parseString("""
DBLE( fac * (0.d0,-1.d0) * (fcoef_d(kh,ih,is1,1,np) * fcoef_d(jh,lh,2,is2,np) ) )
""")[0].cStr())
#sys.exit()

#print(translator.number.parseString("0.0d0")[0].cStr())
#print(translator.number.parseString("0.d0")[0].cStr())
#print(translator.arithmeticExpression.parseString("(b+c)")[0].cStr())
#print(translator.arithmeticExpression.parseString("0.0d0")[0].cStr())
#print(translator.arithmeticExpression.parseString("0.d0+5")[0].cStr())
#print(translator.complexArithmeticExpression.parseString("(b+c,b-c)"))
#print(translator.complexArithmeticExpression.parseString("(0, 0)"))
#print(translator.complexArithmeticExpression.parseString("(0_DP, 0_DP)"))
#print(translator.complexArithmeticExpression.parseString("(0.0_DP, 0.0_DP)"))
#print(translator.complexArithmeticExpression.parseString("(0.d0, 0.d0)"))
#print(translator.conversions.parseString("CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP)")[0].cStr())
#print(translator.arithmeticExpression.parseString("CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP)")[0].cStr())
#print(translator.arithmeticExpression.parseString("1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP)")[0].cStr())
#print(translator.arithmeticExpression.parseString("wfcatom_d(ig,ipol,ibnd) * ( 1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP))")[0].cStr())
#print(translator.assignment.parseString("wfcatom_d(ig,ipol,ibnd) = wfcatom_d(ig,ipol,ibnd) * ( 1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP))")[0].cStr())
#print(translator.arithmeticExpression.parseString("sendsiz + (cuf_i-1)*nppx + i*nppx*ncpx")[0].cStr())
#print(translator.assignment.parseString("it = ( ip - 1 ) * sendsiz + (cuf_i-1)*nppx + i*nppx*ncpx")[0].cStr())
#print(translator.assignment.parseString("it = it0 + (i-1)*nr3px + j*ncpx*nr3px")[0].cStr())
#print(translator.assignment.parseString("mc = desc_ismap_d( i + ioff ) ! this is  m1+(m2-1)*nr1x  of the  current pencil")[0].cStr())
#print(translator.assignment.parseString("m1 = mod (mc-1,nr1x) + 1 ; m2 = (mc-1)/nr1x + 1")[0].cStr())
#print(translator.assignment.parseString("i1 = m2 + ( ir1p__d(m1) - 1 ) * nr2x + (k-1)*(nr2x * my_nr1p_)")[0].cStr())
#print(translator.assignment.parseString("it = it0 + (i-1)*nr3px + j*ncpx*nr3px !desc%nnr !aux*nr3px")[0].cStr())
#print(translator.assignment.parseString("mc = desc_ismap_d( i + ioff ) ! this is  m1+(m2-1)*nr1x  of the  current pencil")[0].cStr())
#print(translator.assignment.parseString("m1 = mod (mc-1,nr1x) + 1 ; m2 = (mc-1)/nr1x + 1")[0].cStr())
#print(translator.assignment.parseString("i1 = m2 + ( ir1p__d(m1) - 1 ) * nr2x + (k-1)*nr2x*my_nr1p_")[0].cStr())
#print(translator.assignment.parseString("f_aux_d( i1 + j*nnr ) = f_in_d( k + it )")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,2) = DBLE( fac * (fcoef_d(kh,ih,is1,1,np) * fcoef_d(jh,lh,2,is2,np) ) )
""")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,2)=becsum_d(ijh,na,2) + DBLE( fac )
""")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,2)=becsum_d(ijh,na,2) + DBLE( fac * (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) + fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ) )
""")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,1)=becsum_d(ijh,na,1) + DBLE( fac * 
     (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,1,is2,np) + 
     fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,2,is2,np)  ) )
""")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,2)=becsum_d(ijh,na,2) + DBLE( fac * 
    (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) +
     fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ) )
""")[0].cStr())
print(translator.assignment.parseString("""
becsum_d(ijh,na,3)=becsum_d(ijh,na,3) + DBLE( fac*(0.d0,-1.d0)*
   (fcoef_d(kh,ih,is1,1,np)*fcoef_d(jh,lh,2,is2,np) - 
    fcoef_d(kh,ih,is1,2,np)*fcoef_d(jh,lh,1,is2,np)  ))
""")[0].cStr())

testdata = []
testdata.append("a3=2.d0")
testdata.append("a3=(2.d0*f1)")
testdata.append("f_d(j+i*nnr_)=(0.0_DP,0.0_DP)")
testdata.append("wfcatom_d(ig,ipol,ibnd) = wfcatom_d(ig,ipol,ibnd) * ( 1.0_DP + 0.05_DP * CMPLX( rr*COS(arg), rr*SIN(arg) ,kind=DP)")
testdata.append("a3=(2.d0*f1-2.d0*f2+(f3+f4))/(z_l-z_r)**3")
testdata.append("a3=(2.d0*f1-2.d0*f2+(f3+f4)*(z_l-z_r))/(z_l-z_r)**3")
testdata.append("f_aux_d(i) = (0.d0, 0.d0)")
testdata.append("f_aux_d(i) = (0, 0)")
testdata.append("f_d(j+i*nnr_) = (0.0_DP,0.0_DP)")
testdata.append("f_d(i) = (0.0_DP,0.0_DP)")

test.run(
   expression     = (translator.complexAssignment | translator.matrixAssignment | translator.assignment),
   testdata       = testdata,
   tag            = None,
   raiseException = True
)
