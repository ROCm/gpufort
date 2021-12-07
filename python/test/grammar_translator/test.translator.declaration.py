#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import os,sys
import time
import unittest
import translator.translator as translator
import utils.logging

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "debug2"
utils.logging.VERBOSE       = False
utils.logging.TRACEBACK     = False
utils.logging.init_logging("log.log",log_format,log_level)

testdata1_types = """
real
real(c_float)
real(8)
real*8
integer
logical
"""

testdata1_pass = """
{type} test
{type} test(:)
{type} test(:,:,:)
{type} test(5)
{type} test(5,6)
{type} test(n)
{type} test(n,k,c)
{type}                  :: test
{type},parameter        :: test
{type},intent(in)       :: test
{type},dimension(:)     :: test
{type}                  :: test(:)
{type},dimension(:,:,:) :: test
{type}                  :: test(:,:,:)
{type},pointer          :: test => null()
{type},pointer          :: test(:) => null()
{type},pointer          :: test(:,:,:) => null()
"""
testdata1_fail = """
{type},parameter     test
{type},intent(in)    test
"""

testdata2_pass = """
logical                  test = .false.
logical                  :: test = .false.
logical,parameter        :: test = .true.
logical,save,parameter   :: test = .false.
"""

testdata3_pass = 
"""
real*8 v(0:*)
real*4 v(0:*)
real*8 v(0:*)
real*4 v(0:*)
integer,  parameter :: bigint = o17777777777
real,parameter :: c0=0.55,ceps=c0**3,blckdr=0.0063,cn=0.75          ,am1=8.0,am2=2.3,am3=35.0,ah1=1.4,ah2=-0.01        ,ah3=1.29,ah4=2.44,ah5=19.8                        ,arimin=0.127,bm1=2.88,bm2=16.0,bh1=3.6,bh2=16.0   ,bh3=720.0,epskm=1.e-3
real, parameter :: prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
integer, parameter:: bigint = o17777777777
real(r8):: coefh(2,4) = reshape(   (/ (/5.46557e+01,-7.30387e-02/),  (/1.09311e+02,-1.46077e-01/),  (/5.11479e+01,-6.82615e-02/),  (/1.02296e+02,-1.36523e-01/) /), (/2,4/) )
real(r8):: coefj(3,2) = reshape(  (/ (/2.82096e-02,2.47836e-04,1.16904e-06/),  (/9.27379e-02,8.04454e-04,6.88844e-06/) /), (/3,2/) )
real(r8):: coefk(3,2) = reshape(  (/ (/2.48852e-01,2.09667e-03,2.60377e-06/) ,  (/1.03594e+00,6.58620e-03,4.04456e-06/) /), (/3,2/) )
real(r8), parameter:: fat(o_fa,nbands) = reshape( (/  (/-1.06665373e-01,  2.90617375e-02, -2.70642049e-04,    1.07595511e-06, -1.97419681e-09,  1.37763374e-12/),  (/ 1.10666537e+00, -2.90617375e-02,  2.70642049e-04,    -1.07595511e-06,  1.97419681e-09, -1.37763374e-12/) /)  , (/o_fa,nbands/) )
real(r8), parameter:: fet(o_fe,nbands) = reshape( (/  (/3.46148163e-01,  1.51240299e-02, -1.21846479e-04,    4.04970123e-07, -6.15368936e-10,  3.52415071e-13/),  (/6.53851837e-01, -1.51240299e-02,  1.21846479e-04,    -4.04970123e-07,  6.15368936e-10, -3.52415071e-13/) /)  , (/o_fa,nbands/) )
real, parameter, private:: ar_volume = 4./3.*pi*(2.5e-6)**3
integer, parameter, dimension(ntemps(1)) :: temps_water=(/ (dtemp(1)*(i-1)+tstart(1),i=1,ntemps(1) )/)
integer, parameter, dimension(ntemps(2)) :: temps_fd=(/(dtemp(2)*(i-1)+tstart(2),i=1,ntemps(2))/)
integer, parameter, dimension(ntemps(3)) :: temps_crystals=(/(dtemp(3)*(i-1)+tstart(3),i=1,ntemps(3))/)
integer, parameter, dimension(ntemps(3)) :: temps_snow=(/(dtemp(3)*(i-1)+tstart(3),i=1,ntemps(3))/)
integer, parameter, dimension(ntemps(4)) :: temps_graupel=(/(dtemp(4)*(i-1)+tstart(4),i=1,ntemps(4))/)
integer, parameter, dimension(ntemps(5)) :: temps_hail=(/(dtemp(5)*(i-1)+tstart(5),i=1,ntemps(5))/)
real, parameter, dimension(nfws(2)) :: fws_fd=(/(1.0/(nfws(2)-1)*(i-1),i=1,nfws(2))/)
real, parameter, dimension(nfws(3)) :: fws_crystals=(/(1.0/(nfws(3)-1)*(i-1),i=1,nfws(3))/)
real, parameter, dimension(nfws(3)) :: fws_snow=(/(1.0/(nfws(3)-1)*(i-1),i=1,nfws(3))/)
real, parameter, dimension(nfws(4)) :: fws_graupel=(/(1.0/(nfws(4)-1)*(i-1),i=1,nfws(4))/)
real, parameter, dimension(nfws(5)) :: fws_hail=(/(1.0/(nfws(5)-1)*(i-1),i=1,nfws(5))/)
real,parameter :: eta_func = ((3.e0/(4.e0*const_pi*const_rho_liq))**2) * const_kappa
real, parameter  :: roqimax = 2.08e22*dimax**8
real, parameter :: xvcmn=0.523599*(2.*cwradn)**3
real, parameter :: xvcmx=0.523599*(2.*xcradmx)**3
real, parameter :: cwmasn5 = 1000.*0.523599*(2.*5.0e-6)**3
real, parameter :: xvimn=0.523599*(2.*5.e-6)**3
real, parameter :: xvimx=0.523599*(2.*1.e-3)**3
real, parameter :: xims=900.*0.523599*(2.*50.e-6)**3
real, parameter :: xims=900.*0.523599*(2.*50.e-6)**3
real, parameter :: xcms=1000.*0.523599*(2.*7.5e-6)**3
real, parameter :: vr3mm = 5.23599e-10*(3.0/1.)**3
real, parameter :: vr4p5mm = 5.23599e-10*(4.5/1.)**3
real, parameter :: d1t = (6.0 * 0.268e-3/(917.* pi))**(1./3.)
real, parameter :: prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
real, parameter :: prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
real, parameter :: lv = 2.54*(10**6)
integer(i_llong),parameter:: lrecl=2**20
integer(i_llong),parameter:: lrecl=2**20
integer(i_llong),parameter:: lrecl=2**20
"""

class TestDeclaration(unittest.TestCase):
    def setUp(self):
        global index
        self._started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def prepare(self,testdata):
        return testdata.strip().split("\n")
    def test1_pass(self):
        testdata = self.prepare(testdata1_pass)
        types    = self.prepare(testdata1_types)
        for statement in testdata:
            for typ in types:
                translator.parse_declaration(statement.format(type=typ))
    def test1_fail(self):
        testdata = self.prepare(testdata1_pass)
        types    = self.prepare(testdata1_types)
        for statement in testdata:
            for typ in types:
                try:
                    translator.parse_declaration(statement.format(type=typ))
                    self.assertTrue(False)
                except Exception as e:
                    return e
    def test2_logicals_pass(self):
        testdata = self.prepare(testdata2_pass)
        for statement in testdata:
            translator.parse_declaration(statement)
    def test3_pass(self):
         
if __name__ == '__main__':
    unittest.main() 
