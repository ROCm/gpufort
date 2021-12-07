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

return_types = ["integer","logical","real"]

testdata1 = """
integer function getrealmpitype()
logical function wrf_dm_on_monitor()
integer function wrf_dm_monitor_rank()
logical function get_poll_servers()
logical function  use_output_servers()
logical function  use_input_servers()
integer function bep_nurbm()
integer function bep_ndm()
integer function bep_nz_um()
integer function bep_ng_u()
integer function bep_nwr_u()
integer function bep_bem_nurbm()
integer function bep_bem_ndm()
integer function bep_bem_nz_um()
integer function bep_bem_ng_u()
integer function bep_bem_nwr_u()
integer function bep_bem_nf_u()
integer function bep_bem_ngb_u()
integer function bep_bem_nbui_max()
integer function get_unused_unit()
real function vv_q()
real function vv_qg()
"""

class TestExpression(unittest.TestCase):
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
