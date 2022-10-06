# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

import enum

import pyparsing

from gpufort import indexer
from gpufort import util

from .. import prepostprocess
from .. import opts

from . import base
from . import traversals
from . import fortran
from . import grammar
    
class Parallelism(enum.Enum):
    UNSPECIFIED = -2
    AUTO = -1
    SEQ=0
    GANG=1
    WORKER=2
    VECTOR=3
    GANG_WORKER=4
    WORKER_VECTOR=5
    GANG_VECTOR=6
    GANG_WORKER_VECTOR=7

    def gang_partitioned_mode(self):
        return self in [Parallelism.GANG,
                        Parallelism.GANG_WORKER,
                        Parallelism.GANG_WORKER_VECTOR]
    
    def worker_partitioned_mode(self):
        return self in [Parallelism.WORKER,
                        Parallelism.GANG_WORKER,
                        Parallelism.GANG_WORKER_VECTOR]
    
    def vector_partitioned_mode(self):
        return self in [Parallelism.VECTOR,
                        Parallelism.GANG_VECTOR,
                        Parallelism.GANG_WORKER_VECTOR]

class IterateOrder(enum.Enum):
    UNSPECIFIED = -2
    AUTO = -1 # requires static analyis
    SEQ = 0
    INDEPENDENT = 1

class IDeviceSpec():
    def applies_to(self,device_type):
        return True

    def num_collapse(self):
        return grammar.CLAUSE_NOT_FOUND

    def tile_sizes(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def grid_expr_fstr(self):
        """ only CUF """
        return None

    def block_expr_fstr(self):
        """ only CUF """
        return None

    def num_gangs(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_threads_in_block(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_workers(self):
        """ only ACC """
        return grammar.CLAUSE_NOT_FOUND

    def vector_length(self):
        return grammar.CLAUSE_NOT_FOUND

    def parallelism(self):
        return Parallelism.AUTO

    def order_of_iterates(self):
        return IterateOrder.AUTO 

# todo: move to different place
def format_directive(directive_line, max_line_width):
    result = ""
    line = ""
    tokens = directive_line.split(" ")
    sentinel = tokens[0]
    for tk in tokens:
        if len(line + tk) > max_line_width - 1:
            result += line + "&\n"
            line = sentinel + " "
        line += tk + " "
    result += line.rstrip()
    return result
