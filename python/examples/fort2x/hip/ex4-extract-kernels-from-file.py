#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os, sys
import addtoplevelpath
from gpufort import util
from gpufort import linemapper
from gpufort import fort2x

import json

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose = False
util.logging.init_logging("log.log", LOG_FORMAT, "warning")

file_content = """\
program main
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(x(1:N),y(1:N))
 
  !$acc parallel loop present(x,y)
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"
end program
"""

codegen, linemaps = fort2x.hip.create_code_generator(file_content=file_content,
                                                     emit_cpu_launcher=True)
codegen.run()

print("modified Fortran file:")
print("```")
print(
    linemapper.modify_file(linemaps,
                           file_content=file_content,
                           ifdef_macro="_GPUFORT"))
                           #ifdef_macro=None))
print("```")

print("main C++ file:")
print("```")
print(codegen.cpp_filegen.generate_code())
print("```")
for path, filegen in codegen.cpp_filegens_per_module:
    print("{}:".format(path))
    print("```")
    print(filegen.generate_code())
    print("```")

#for modulegen in codegen.fortran_modulegens:
#    modulegen.used_modules.append({"name" : "iso_c_binding", "only" : []})
#    modulegen.used_modules.append({"name" : "gpufort_array", "only" : []})
#    modulegen.used_modules.append({"name" : "hipfort", "only" : []})
#    modulegen.used_modules.append({"name" : "hipfort_check", "only" : []})
#    print("Module {}:".format(modulegen.name))
#    print("```")
#    print(modulegen.generate_code())
#    print("```")
