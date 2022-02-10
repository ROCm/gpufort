#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
from gpufort import util
from gpufort import fort2x

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

declaration_list= """\
type inner
  real(8)            :: scalar_double
  integer(4),pointer :: array_integer(:,:)
end type inner

type outer
  type(inner)                            :: single_inner
  type(inner),allocatable,dimension(:,:) :: array_inner
#ifdef MORE_FIELDS
  integer(4),pointer                     :: array_integer01(:,:,:)
  integer(4),pointer                     :: array_integer02(:,:,:)
  integer(4),pointer                     :: array_integer03(:,:,:)
  integer(4),pointer                     :: array_integer04(:,:,:)
  integer(4),pointer                     :: array_integer05(:,:,:)
  integer(4),pointer                     :: array_integer06(:,:,:)
  integer(4),pointer                     :: array_integer07(:,:,:)
  integer(4),pointer                     :: array_integer08(:,:,:)
  integer(4),pointer                     :: array_integer09(:,:,:)
  integer(4),pointer                     :: array_integer10(:,:,:)
  integer(4),pointer                     :: array_integer11(:,:,:)
  integer(4),pointer                     :: array_integer12(:,:,:)
#endif
end type outer
"""

used_modules = [{"name" : mod, "only" : []} for mod in [
                                                       "iso_c_binding",
                                                       "hipfort_check",
                                                       "hipfort",
                                                       "gpufort_array",
                                                       ]]

derivedtypegen = fort2x.hip.create_derived_type_generator(declaration_list,
                                                          used_modules=used_modules,
                                                          preproc_options="")
                                                          #preproc_options="-DMORE_FIELDS")
print("# Interoperable Fortran derived types:\n")
print("\n".join(derivedtypegen.render_derived_type_definitions_f03()))
print("\n# Copy routines for creating interoperable from original type:\n")
print("\n".join(derivedtypegen.render_derived_type_routines_f03()))
print("\n# C++ definitions:\n")
print("\n".join(derivedtypegen.render_derived_type_definitions_cpp()))