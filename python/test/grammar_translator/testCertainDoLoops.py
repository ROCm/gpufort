#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator

testdata = []
testdata.append("""
DO i = 1, desc%my_nr3p*my_nr1p_*nr2x
  f_aux_d(i) = (0.d0, 0.d0)
END DO
""")
testdata.append("""
DO k = 1, desc%my_nr3p
   it = it0 + (i-1)*nr3px
   mc = desc_ismap_d( i + ioff ) ! this is  m1+(m2-1)*nr1x  of the  current pencil
   m1 = mod (mc-1,nr1x) + 1 ; m2 = (mc-1)/nr1x + 1
   i1 = m2 + ( ir1p__d(m1) - 1 ) * nr2x + (k-1)*nr2x*my_nr1p_

   f_aux_d( i1 ) = f_in_d( k + it )
   !i1 = i1 + desc%nr2x*my_nr1p_
ENDDO
""")
testdata.append("""
do lm = 1, lpx_d (ivl, jvl)
  lp = lpl_d (ivl, jvl, lm)
  if (lp == 1) then
     l = 1
     sig = 1.0d0
     ind = 1
  else if ( lp <= 4) then
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
""")

for i,v in enumerate(testdata):
    vMod = translator.prepareFortranSnippet(v)
    result = translator.doLoop.parseString(vMod)
    print("{}:\n{}".format(i,result[0].c_str()))

#test.run(
#   expression     = translator.ifElseBlock,
#   testdata       = testdata,
#   tag            = None,
#   raiseException = True
#)