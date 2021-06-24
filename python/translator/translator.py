# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
import os,sys,traceback
import logging
import collections
import ast
import re

# recursive inclusion
import indexer.scoper as scoper
import utils.logging
import utils.pyparsingutils 

#from grammar import *
CASELESS    = True
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_options.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_base.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_f03.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_directives.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_cuf.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_acc.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_api.py.in")).read())
