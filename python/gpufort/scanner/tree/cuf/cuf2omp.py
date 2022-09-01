# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys

from gpufort import indexer
from gpufort import translator
from gpufort import util

from ... import opts

from .. import backends
from .. import nodes

from . import cufnodes
from . import cuf2hip

dest_dialects = ["omp"]
backends.supported_destination_dialects.add("omp")

def CufLoopNest2Omp(stcomputeconstruct,
                    joined_lines,
                    joined_statements,
                    statements_fully_cover_lines,
                    index=[]):
    """Analyze based on statements but modify original lines if these are
    fully covered by the statements.
    """
    if not opts.translate_compute_constructs:
        return (None,False)
    parent_tag = stcomputeconstruct.parent.tag()
    scope      = indexer.scope.create_scope(index, parent_tag)
    ttcomputeconstruct = stcomputeconstruct.parse_result 
    
    arrays       = translator.analysis.arrays_in_subtree(ttcomputeconstruct, scope)
    inout_arrays = translator.analysis.inout_arrays_in_subtree(ttcomputeconstruct, scope)

    snippet = joined_lines if statements_fully_cover_lines else joined_statements
    return translator.codegen.translate_compute_construct_to_omp(snippet, ttcomputeconstruct, inout_arrays, arrays), True

cufnodes.STCufLoopNest.register_backend(dest_dialects,CufLoopNest2Omp)

# Current strategy: Convert to HIP instead of OpenMP for these operations
nodes.STAllocate.register_backend("cuf", dest_dialects, cuf2hip.handle_allocate_cuf)
nodes.STDeallocate.register_backend("cuf", dest_dialects, cuf2hip.handle_deallocate_cuf)
nodes.STUseStatement.register_backend("cuf", dest_dialects, cuf2hip.handle_use_statement_cuf)

# Postprorcess backend
backends.register_postprocess_backend("cuf", dest_dialects, cuf2hip.postprocess_tree_cuf)