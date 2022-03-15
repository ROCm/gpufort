#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
from gpufort import util
from gpufort import fort2x

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose = False
util.logging.init_logging("log.log", LOG_FORMAT, "debug")

declaration_list = """\
integer, parameter :: N = 1000, M=2000
integer :: i,j,k
integer(4) :: x(N), y(N), y_exact(N)
"""

annotated_loop_nest = """\
!$acc parallel loop present(x,y) collapse(2)
do j = 1, M
  do i = 1, N
    x(i,j) = 1
    y(i,j) = 2

    do while ( k < 10 )
      y(i,j) = x(i,j) * k
      k = k + 1
    end do
  end do
end do
"""

#print(ttloopnest.c_str())
kernelgen = fort2x.hip.create_kernel_generator_from_loop_nest(
    declaration_list, annotated_loop_nest, kernel_name="mykernel")

print("\n".join(kernelgen.render_gpu_kernel_cpp()))
launcher = kernelgen.create_launcher_context(kind="hip",
                                             debug_code=False,
                                             used_modules=[])
print("\n".join(kernelgen.render_gpu_launcher_cpp(launcher)))
launcher = kernelgen.create_launcher_context(kind="hip_ps",
                                             debug_code=False,
                                             used_modules=[])
print("\n".join(kernelgen.render_gpu_launcher_cpp(launcher)))
