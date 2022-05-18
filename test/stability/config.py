# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
opts.profiling_enable = False

def default_parameter_filter(scope_tag,f_type,f_kind,name,rank):
    return True
def default_prepend_callback(scope_tag):
    """:return C++ code lines to prepend to the
    body of the namespace mapped to the scope tag."""
    return []
fort2x.namespacegen.opts.parameter_filter= default_parameter_filter
        # Selector for parameter types that should be mapped to the C++ namespace derived
        # from the scope with the given tag.
fort2x.namespacegen.opts.prepend_callback= default_prepend_callback
        # If resolve_all_parameters_via_compiler is False, C++ code lines to prepend to the
        # body of the namespace derived from the scope tag. Otherwise, must return Fortran statements
        # to preprend.
fort2x.namespacegen.opts.comment_body= False
        # If all translated parameter declarations that are rendered into the namespace
        # should be commented out. This allows to copy certain parameters
        # and put them into the lists returned by the prepend/append callback for a given scope tag.
        # Only considered if resolve_all_parameters_via_compiler is False.
fort2x.namespacegen.opts.resolve_all_parameters_via_compiler= True
        # Resolve all parameters via a Fortran compiler.
fort2x.namespacegen.opts.fortran_compiler= os.environ.get("FC","gfortran")
        # Compiler to use for resolving parameters via (Fortran) compiler.
fort2x.namespacegen.opts.fortran_compiler_flags= os.environ.get("FCFLAGS","")
        # Compiler flags to use for resolving parameters via (Fortran) compiler.
