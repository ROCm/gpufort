# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .base import *
from .fortran import *
from .acc import *
from .cuf import *
from .directives import *
from . import traversals

from gpufort import grammar as _grammar
grammar = _grammar.Grammar(
  ignorecase=True,
  #unary_op_parse_action=TTUnaryOp,   
  #binary_op_parse_action=TTBinaryOp
)
del _grammar    

#set_fortran_parse_actions(grammar)
#set_acc_parse_actions(grammar)
#set_cuf_parse_actions(grammar)
