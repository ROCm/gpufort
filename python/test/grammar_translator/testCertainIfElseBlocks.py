#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator

#print(_else_if_statement.parseString("""
#elseif ( lp < 16 ) then
#    phi  = atan( g(2,ig)/g(1,ig) ) + pi
#"""))

testdata = []
testdata.append("""
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
    else if ( lp <= 16 ) then
       l = 4
        sig = 1.0d0
       ind = 2
    else if ( lp <= 25 ) then
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
""")
testdata.append("""
if (lmax == 0) then
  ylm(ig,1) =  sqrt (1.d0 / fpi)
elseif (g(1,ig) < -eps) then
  return
else
  return
end if
""")
testdata.append("""
if (g(1,ig) > eps) then
  phi  = atan( g(2,ig)/g(1,ig) )
elseif (g(1,ig) < -eps) then
  phi  = atan( g(2,ig)/g(1,ig) ) + pi
else
  phi  = sign( pi/2.d0,g(2,ig) )
end if
""")
testdata.append("""if (lmax == 0) then
if (lmax == 0) then
    ! phi  = atan( g(2,ig)/g(1,ig) )
  end if
end if""")
testdata.append("""
if (ig <= ng) then
  if (g(1,ig) > eps) then
    phi  = atan( g(2,ig)/g(1,ig) )
  elseif (g(1,ig) < -eps) then
    phi  = atan( g(2,ig)/g(1,ig) ) + pi
  else
    phi  = sign( pi/2.d0,g(2,ig) )
  end if
  lm = 4
  do l = 2, lmax
  ! asdasd
  end do
end if
""")
# 4
testdata.append("""
if (lp == 1) then
   l = 1
elseif ( lp < 16 ) then
   l = 1
else
   l = 7
   sig =-1.0d0
   ind = 1
endif
""")
testdata.append("""
if (lp == 1) then
   l = 1
   sig = 1.0d0
   ind = 1
elseif ( lp < 16 ) then
   !l = 4
   ! sig = 1.0d0
   !ind = 2
else
   l = 7
   sig =-1.0d0
   ind = 1
endif
""")
testdata.append("""
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
""")
testdata.append("""
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
""")

for i,v in enumerate(testdata):
    vMod = translator.prepareFortranSnippet(v)
    result = translator.ifElseBlock.parseString(vMod)
    #result = translator.block.parseString(vMod)
    print("{}:\n{}".format(i,translator.make_c_str(result)))

#test.run(
#   expression     = translator.ifElseBlock,
#   testdata       = testdata,
#   tag            = None,
#   raiseException = True
#)