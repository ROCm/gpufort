# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import indexer
from gpufort import linemapper
from gpufort import linemapper
from gpufort import translator
from gpufort import scanner
from gpufort import util
from . import hipkernelgen
from . import hipderivedtypegen
from . import hipcodegen

__all__ = [
          "create_kernel_generator_from_loop_nest",
          "create_derived_type_generator",
          "create_code_generator",
          ]

def create_kernel_generator_from_loop_nest(declaration_list_snippet,
                                           loop_nest_snippet,
                                           **kwargs):
    r"""Create HIP kernel generator from a declaration list snippet and
    the snippet of an directive-annotated loop nest.
    :return a code generator that can generate a HIP kernel
            and a number of different kernel launchers based on
            the original input
    :rtype: hipkernelgen.HipKernelGenerator4LoopNest
    :param str declaration_list_snippet: A Fortran declaration list, i.e. a number of Fortran
                                         variable and derived type declarations.
    :param str loop_nest_snippet: A Fortran loop nest annotated with directives (CUDA Fortran, OpenACC).
    :param \*\*kwargs: See below. 
    
    :Keyword Arguments:
 
    * *preproc_options* (`str`):
        C-style preprocessor options [default: '']
    * *kernel_name* (`str`):
        Name to give the kernel.
    * *kernel_hash* (`str`):
        Hash code encoding the significant kernel content (expressions and directives).
    """
    preproc_options    = util.kwargs.get_value("preproc_options","",**kwargs)
    scope              = indexer.scope.create_scope_from_declaration_list(declaration_list_snippet,
                                                                    preproc_options)
    linemaps           = linemapper.read_lines(loop_nest_snippet.splitlines(keepends=True),
                                               preproc_options)
    fortran_statements = linemapper.get_statement_bodies(linemaps)
    ttloopnest         = translator.parse_loop_kernel(fortran_statements,
                                                      scope)

    return hipkernelgen.HipKernelGenerator4LoopNest(ttloopnest,
                                                 scope,
                                                 fortran_snippet="\n".join(fortran_statements),
                                                 **kwargs)

def create_derived_type_generator(declaration_list_snippet,
                                  used_modules=[],
                                  preproc_options=""):
    """Create an interoperable derived type generator from a declaration list snippet
       that describes the types.
    :return a code generator that can generate interoperable types
            from a declaration list plus routines for copying
             
    :rtype: hipderivedtypegen.DerivedTypeGenerator
    :param str declaration_list_snippet: A Fortran declaration list, i.e. a number of Fortran
                                         variable and derived type declarations.
    :param list used_modules: List of dicts with keys 'name' (str) and 'only' (list of str)
    :param str preproc_options: C-style preprocessor options
    """
    scope = indexer.scope.create_scope_from_declaration_list(declaration_list_snippet,
                                                        preproc_options)
    return hipderivedtypegen.HipDerivedTypeGenerator(scope["types"],
                                                     used_modules)

def create_code_generator(**kwargs):
    r"""Create HIP code generator from an input file and its dependencies.
    :param \*\*kw_args: See below.

    :Keyword Arguments:

    * *file_path* (``str``):
        Path to the file that should be parsed.
    * *file_content* (``str``):
        Content of the file that should be parsed.
    * *file_linemaps* (``list``):
        Linemaps of the file that should be parsed;
        see GPUFORT's linemapper component.
    * *file_is_indexed* (``bool``):
        Index already contains entries for the main file.
    * *preproc_options* (``str``): Options to pass to the C preprocessor
        of the linemapper component.
    * *other_files_paths* (``list(str)``):
        Paths to other files that contain module files that
        contain definitions required for parsing the main file.
        NOTE: It is assumed that these files have not been indexed yet.
    * *other_files_contents* (``list(str)``):
        Content that contain module files that
        contain definitions required for parsing the main file.
        NOTE: It is assumed that these files have not been indexed yet.
    * *index* (``list``):
        Index records created via GPUFORT's indexer component.
    """ 
    stree, index, linemaps = scanner.parse_file(**kwargs)
    return hipcodegen.HipCodeGenerator(stree,index,**kwargs), linemaps
