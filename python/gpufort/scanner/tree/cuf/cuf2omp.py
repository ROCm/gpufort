# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys

from gpufort import translator
from gpufort import util

from ... import opts

from .. import backends
from .. import nodes

from . import cufnodes
from . import cuf2hip

dest_dialects = ["omp"]
backends.supported_destination_dialects.add("omp")

def CufLoopNest2Omp(stloopnest,*args,**kwargs):
    """Analyze based on statements but modify original lines if these are
    fully covered by the statements.
    """
    try:
        parent_tag = stloopnest.parent.tag()
        scope = indexer.scope.create_scope(index, parent_tag)
        parse_result = translator.parse_loop_kernel(
            joined_statements.splitlines(), scope)
        f_snippet = joined_lines if statements_fully_cover_lines else joined_statements
        return parse_result.omp_f_str(f_snippet), True
    except Exception as e:
        util.logging.log_exception(opts.log_prefix, "CufLoopNest2Omp",
                                   "failed to parse loop kernel")
        sys.exit(2) # TODO error code

cufnodes.STCufLoopNest.register_backend(dest_dialects,CufLoopNest2Omp)

# Current strategy: Convert to HIP instead of OpenMP for these operations
nodes.STAllocate.register_backend("cuf", dest_dialects, cuf2hip.handle_allocate_cuf)
nodes.STDeallocate.register_backend("cuf", dest_dialects, cuf2hip.handle_deallocate_cuf)
nodes.STUseStatement.register_backend("cuf", dest_dialects, cuf2hip.handle_use_statement_cuf)

# Postprorcess backend
backends.register_postprocess_backend("cuf", dest_dialects, cuf2hip.postprocess_tree_cuf)
