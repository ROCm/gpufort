# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

from gpufort import util

from .. import tree
from .. import optvals

class CufKernelDoInfo:
    def __init__(self):
        self.num_loops = optvals.OptionalSingleValue()
        self.grid = optvals.OptionalSingleValue()
        self.block = optvals.OptionalSingleValue()
        self.sharedmem = optvals.OptionalSingleValue()
        self.stream = optvals.OptionalSingleValue()
        self.reduction = optvals.OptionalDictValue()

def _analyze_directive_action(ttnode,parents,result):
    if isinstance(ttnode,TTCufKernelDoArgNumLoops):
        result.num_loops = ttnode.value
    elif isinstance(ttnode,tree.TTCufKernelDoArgGrid):
        result.grid.value = ttnode.value
    elif isinstance(ttnode,tree.TTCufKernelDoArgBlock):
        result.block.value = ttnode.value
    elif isinstance(ttnode,tree.TTCufKernelDoArgSharedmem):
        result.sharedmem.value = ttnode.value
    elif isinstance(ttnode,tree.TTCufKernelDoArgStream):
        result.stream.value = ttnode.value

def analyze_directive(ttcufkerneldo):
    result = CufKernelDoInfo()
    tree.traversals.traverse(
        ttcufkerneldo,
        _analyze_directive_action,
        tree.traversals.no_action,
        tree.traversals.no_crit,
        result)
    return result

def _visit_directive_clause(expr,parents,lvalues,rvalues):
    """Traversal action that searches through arguments of loop clauses and discover
    rvalues expressions."""
    if isinstance(expr,(
        tree.TTCufKernelDoArgNumLoops,
        tree.TTCufKernelDoArgGrid,
        tree.TTCufKernelDoArgBlock,
        tree.TTCufKernelDoArgSharedmem,
        tree.TTCufKernelDoArgStream)):
          _find_lvalues_and_rvalues(expr,lvalues,rvalues)

def find_lvalues_and_rvalues_in_directive(ttcufkerneldo):
    """Search through arguments of loop directive clauses and discover
    rvalues expressions."""
    tree.traversals.traverse(
        ttcufkerneldo,
        _visit_directive_clause,
        tree.traversals.no_action,
        tree.traversals.no_crit,
        lvalues,
        rvalues)
