#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
import linemapper.linemapper as linemapper
import linemapper.linemapperutils as linemapperutils
import indexer.indexer as indexer
import indexer.indexerutils as indexerutils 
import translator.translator as translator
import utils.logging
import fort2x.hip.kernelgen

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

declaration_list= """\
integer, parameter :: N = 1000
integer :: i
integer(4) :: x(N), y(N), y_exact(N)
"""

annotated_loop_nest = """\
!$acc parallel loop present(x,y)
do i = 1, N
  x(i) = 1
  y(i) = 2
end do
"""  

# TODO simplify
scope              = indexerutils.create_scope_from_declaration_list(declaration_list)
linemaps           = linemapper.read_lines(annotated_loop_nest.split("\n"))
fortran_statements = linemapperutils.get_statement_bodies(linemaps)
ttloopnest         = translator.parse_loop_kernel(fortran_statements,
                                                  scope)
#print(ttloopnest.c_str())
kernelgen = fort2x.hip.kernelgen.HipKernelGenerator4LoopNest("mykernel",
                                                             "abcdefgh",
                                                             ttloopnest,
                                                             scope,
                                                             "\n".join(fortran_statements))

print("\n".join(kernelgen.render_gpu_kernel_cpp()))
launcher = kernelgen.create_launcher_context(kind="hip",
                                                  debug_output=False,
                                                  used_modules=[])
print("\n".join(kernelgen.render_gpu_launcher_cpp(launcher)))
