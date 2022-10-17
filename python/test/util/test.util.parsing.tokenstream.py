#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
from gpufort import util

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

class TestTokenStream(unittest.TestCase):
    def prepare(self,text):
        return text.strip().splitlines()
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_00_test_token_stream_getitem_len_keepws_ignorews(self):
        test = "ident0 ident1 ident2 ident3" 
        # pop 1
        stream = util.parsing.TokenStream(
          test,
          keepws=True,
          ignorews=True,
        )
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front()
        self.assertEqual(stream[0],"ident1")
        self.assertEqual(stream[1],"ident2")
        self.assertEqual(stream[2],"ident3")
        self.assertEqual(len(stream),3)
        stream.pop_front()
        self.assertEqual(stream[0],"ident2")
        self.assertEqual(stream[1],"ident3")
        self.assertEqual(len(stream),2)
        stream.pop_front()
        self.assertEqual(stream[0],"ident3")
        self.assertEqual(len(stream),1)
        stream.pop_front()
        self.assertEqual(len(stream),0)
        # pop 2
        stream.current_front = 0 
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front(2)
        self.assertEqual(stream[0],"ident2")
        self.assertEqual(stream[1],"ident3")
        self.assertEqual(len(stream),2)
        stream.pop_front(2)
        self.assertEqual(len(stream),0)
        # pop 4
        stream.current_front = 0 
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front(4)
        self.assertEqual(len(stream),0)
    def test_01_test_token_stream_getitem_len_keepws(self):
        test = "ident0 ident1 ident2 ident3" 
        # pop 1
        stream = util.parsing.TokenStream(
          test,
          keepws=True,
          ignorews=False,
        )
        self.assertEqual(stream[2*0],"ident0")
        self.assertEqual(stream[2*1],"ident1")
        self.assertEqual(stream[2*2],"ident2")
        self.assertEqual(stream[2*3],"ident3")
        self.assertTrue(stream[2*0+1].isspace())
        self.assertTrue(stream[2*1+1].isspace())
        self.assertTrue(stream[2*2+1].isspace())
        self.assertEqual(len(stream),7)
        # pop 4
        stream.pop_front(4)
        self.assertEqual(stream[2*0],"ident2")
        self.assertEqual(stream[2*1],"ident3")
        self.assertTrue(stream[2*0+1].isspace())
        self.assertEqual(len(stream),3)
    def test_02_test_token_stream_getitem_len(self):
        test = "ident0 ident1 ident2 ident3" 
        # pop 1
        stream = util.parsing.TokenStream(
          test,
          keepws=False,
          ignorews=False, # or True
        )
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front()
        self.assertEqual(stream[0],"ident1")
        self.assertEqual(stream[1],"ident2")
        self.assertEqual(stream[2],"ident3")
        self.assertEqual(len(stream),3)
        stream.pop_front()
        self.assertEqual(stream[0],"ident2")
        self.assertEqual(stream[1],"ident3")
        self.assertEqual(len(stream),2)
        stream.pop_front()
        self.assertEqual(stream[0],"ident3")
        self.assertEqual(len(stream),1)
        stream.pop_front()
        self.assertEqual(len(stream),0)
        # pop 2
        stream.current_front = 0 
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front(2)
        self.assertEqual(stream[0],"ident2")
        self.assertEqual(stream[1],"ident3")
        self.assertEqual(len(stream),2)
        stream.pop_front(2)
        self.assertEqual(len(stream),0)
        # pop 4
        stream.current_front = 0 
        self.assertEqual(stream[0],"ident0")
        self.assertEqual(stream[1],"ident1")
        self.assertEqual(stream[2],"ident2")
        self.assertEqual(stream[3],"ident3")
        self.assertEqual(len(stream),4)
        stream.pop_front(4)
        self.assertEqual(len(stream),0)
 
if __name__ == '__main__':
    unittest.main() 
