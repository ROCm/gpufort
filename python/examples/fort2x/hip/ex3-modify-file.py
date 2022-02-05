#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import addtoplevelpath
import utils.logging
import fort2x.hip.fort2hiputils as fort2hiputils
import linemapper.linemapper as linemapper

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

file_content= """\
module mymod
  type inner
    real(8)            :: scalar_double
    integer(4),pointer :: array_integer(:,:)
  end type inner

  type outer
    type(inner)                            :: single_inner
    type(inner),allocatable,dimension(:,:) :: array_inner
    integer(4),pointer                     :: array_integer(:,:,:)
  end type outer
end module
"""

codegen, linemaps = fort2hiputils.create_code_generator(file_content=file_content)
codegen.run()

stree = codegen.stree
sys.exit()
print("modified Fortran file:")
print("```")
print(linemapper.modify_file(linemaps,file_content=file_content))
print("```")

print("main C++ file:")
print("```")
print(codegen.cpp_filegen.generate_code())
print("```")
for path,filegen in codegen.cpp_filegens_per_module:
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
