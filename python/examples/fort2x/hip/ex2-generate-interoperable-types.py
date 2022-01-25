#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
import utils.logging
import fort2x.hip.fort2hiputils as fort2hiputils

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

declaration_list= """\
type inner
  real(8)            :: scalar_double
  integer(4),pointer :: array_integer(:,:)
end type inner

type outer
  type(inner)                            :: single_inner
  type(inner),allocatable,dimension(:,:) :: array_inner
  integer(4),pointer                      :: array_integer(:,:,:)
end type outer
"""

used_modules = [{"name" : mod, "only" : []} for mod in [
                                                       "iso_c_binding",
                                                       "hipfort_check",
                                                       "gpufort_array",
                                                       ]]

derivedtypegen = fort2hiputils.create_interoperable_derived_type_generator(declaration_list,
                                                                           used_modules=used_modules,
                                                                           preproc_options="")
print("# Interoperable Fortran derived types:\n")
print("\n".join(derivedtypegen.render_derived_type_definitions_f03()))
print("\n# Copy routines for creating interoperable from original type:\n")
print("\n".join(derivedtypegen.render_derived_type_routines_f03()))
print("\n# C++ definitions:\n")
print("\n".join(derivedtypegen.render_derived_type_definitions_cpp()))
