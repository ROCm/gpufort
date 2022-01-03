# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
exec(open(os.path.join(GRAMMAR_DIR,"grammar.py")).read())

TRANSLATOR_DIR = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(TRANSLATOR_DIR, "translator_options.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_base.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_f03.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_directives.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_cuf.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_acc.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_parser.py.in")).read())
exec(open(os.path.join(TRANSLATOR_DIR, "translator_api.py.in")).read())