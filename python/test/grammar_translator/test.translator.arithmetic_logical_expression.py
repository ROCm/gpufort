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

testdata1 = """
ceps=c0**3
prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
ar_volume = 4./3.*pi*(2.5e-6)**3
eta_func = ((3.e0/(4.e0*const_pi*const_rho_liq))**2) * const_kappa
roqimax = 2.08e22*dimax**8
xvcmn=0.523599*(2.*cwradn)**3
xvcmx=0.523599*(2.*xcradmx)**3
cwmasn5 = 1000.*0.523599*(2.*5.0e-6)**3
xvimn=0.523599*(2.*5.e-6)**3
xvimx=0.523599*(2.*1.e-3)**3
xims=900.*0.523599*(2.*50.e-6)**3
xims=900.*0.523599*(2.*50.e-6)**3
xcms=1000.*0.523599*(2.*7.5e-6)**3
vr3mm = 5.23599e-10*(3.0/1.)**3
vr4p5mm = 5.23599e-10*(4.5/1.)**3
d1t = (6.0 * 0.268e-3/(917.* pi))**(1./3.)
prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
prt0 = cphi/ceps * ftau0**2 / 2. / fth0**2
lv = 2.54*(10**6)
lrecl=2**20
lrecl=2**20
lrecl=2**20
"""

class TestArithmeticLogicalExpression(unittest.TestCase):
    def setUp(self):
        global index
        self._started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def prepare(self,testdata):
        return testdata.strip().split("\n")
if __name__ == '__main__':
    unittest.main() 
