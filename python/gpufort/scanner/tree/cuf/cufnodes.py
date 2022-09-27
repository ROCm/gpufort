# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap
import re

from gpufort import util
from gpufort import translator
from gpufort import indexer

from .. import nodes
from .. import opts
from .. import grammar

class STCufDirective(nodes.STDirective):

    """This class has the functionality of a kernel if the stored lines contain
    a cuf kernel directivity. 
    """
    def __init__(self, first_linemap, first_linemap_first_statement):
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   sentinel="!$cuf")

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        assert False, "Currently, there are only CUF parallel directives"

class STCufLoopNest(STCufDirective, nodes.STComputeConstruct):
    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, func):
        cls._backends.append((dest_dialects, func))

    def __init__(self, first_linemap, first_linemap_first_statement):
        STCufDirective.__init__(self, first_linemap, first_linemap_first_statement)
        nodes.STComputeConstruct.__init__(self, first_linemap, first_linemap_first_statement)
        self.dest_dialect = opts.destination_dialect

    def transform(self,joined_lines,joined_statements,*args,**kwargs):
        if not opts.translate_compute_constructs:
            return (None, False)
        result = joined_statements
        transformed = False
        for dest_dialects, func in self.__class__._backends:
            if self.dest_dialect in dest_dialects:
                result, transformed1 = func(self, joined_lines, joined_statements, *args, **kwargs)
                transformed = transformed or transformed1
        return result, transformed

class STCufLibCall(nodes.STNode):

    def __init__(self, first_linemap, first_linemap_first_statement):
        nodes.STNode.__init__(self, first_linemap,
                              first_linemap_first_statement)
        self._cuda_api = ""
        self._has_cublas = False

    def has_cublas(self):
        """
        :return: Scanned code lines contain a cublas call.
        :rtype: bool
        """
        return self._has_cublas

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        snippet = joined_statements
        transformed = False
        old_snippet = snippet
        indent = self.first_line_indent()
        if not opts.keep_cuda_lib_names:

            def repl_memcpy(parse_result):
                dest_name = parse_result.dest_name_fstr()
                src_name = parse_result.src_name_fstr()
                dest_indexed_var = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                  dest_name)
                src_indexed_var  = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                  src_name)
                dest_on_device = index_var_is_on_device(dest_indexed_var)
                src_on_device = index_var_is_on_device(src_indexed_var)
                subst = parse_result.hip_fstr(dest_on_device, src_on_device)
                return (subst, True)

            snippet, _ = util.pyparsing.replace_all(
                snippet, translator.tree.grammar.cuf_cudamemcpy_variant,
                repl_memcpy)

        def repl_cublas(parse_result):
            subst = parse_result.fstr(indent)
            return (subst, True)

        snippet, have_cublas = util.pyparsing.replace_all(
            snippet, translator.tree.grammar.cuf_cublas_call, repl_cublas)
        if have_cublas:
            self._has_cublas = True
        for elem in grammar.CUDA_RUNTIME_ENUMS:
            snippet = re.sub(elem,elem.replace("cuda", "hip").replace("CUDA", "HIP"), snippet, re.IGNORECASE)
        for elem in grammar.CUDA_LIB_ENUMS:
            snippet = re.sub(elem,elem.replace("cuda", "hip").replace("CUDA", "HIP"), snippet, re.IGNORECASE)
        for elem in grammar.ALL_HOST_ROUTINES: # runtime routines
            snippet = re.sub(elem,elem.replace("cuda", "hip").replace("CUDA", "HIP"), snippet, re.IGNORECASE)
        for elem in grammar.CUDA_MATH_LIB_FUNCTIONS:
            snippet = re.sub(elem,elem.replace("cuda", "hip").replace("CUDA", "HIP"), snippet, re.IGNORECASE)
        transformed = snippet.lower() != old_snippet
        return snippet, transformed


class STCufKernelCall(nodes.STNode):
    """Converts CUDA Fortran kernel calls to kernel launcher call."""

    # TODO add wrapper call
    # TODO ensure use gpufort_array is added to one of the parent nodes
    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if opts.destination_dialect.startswith("hip"):
            self.parent.add_use_statement("gpufort_array")
            self.parent.add_use_statement("hipfort_types",only=["dim3"])
            #
            kernel_name, launch_params, call_args = util.parsing.parse_cuf_kernel_call(joined_statements)
            iprocedure = indexer.scope.search_index_for_procedure(index,self.parent.tag(),kernel_name)
            if len(call_args) != len(iprocedure["dummy_args"]):
                raise util.error.SyntaxError("kernel subroutine '{}' expects {} arguments but {} were passed".format
                        (kernel_name,len(iprocedure["dummy_args"]),len(call_args)))
            for i,arg in enumerate(iprocedure["dummy_args"]):
                ivar = next((ivar for ivar in iprocedure["variables"] if ivar["name"] == arg),None)
                if ivar == None:
                    raise util.error.LookupError("could not find index record for dummy argument '{}'".format(arg))
                if ivar["rank"] > 0:
                    expr = call_args[i]
                    call_args[i] = "".join(["gpufort_array",str(ivar["rank"]),"_wrap_device_cptr(&\n    ",
                                   "c_loc({0}),shape({0},kind=c_int),lbound({0},kind=c_int))".format(expr,ivar["rank"])])
            if len(launch_params) == 4:
                if launch_params[3]=="0": launch_params[3] = "c_null_ptr" # stream
            else:
                launch_params += ["0","c_null_ptr"] # sharedmem, stream
            launch_params.append(".true.") # async
            tokens = ["call launch_",kernel_name,"_hip(&\n  ",",&\n  ".join(launch_params+call_args),")"]
            indent = self.first_line_indent()
            return textwrap.indent("".join(tokens),indent), True
        else:
            return "", False

class STCufAttributes(nodes.STNode):
    """CUDA Fortran specific intrinsic that needs to be removed/commented out
    in any case.
    """
    def transform(self,*args,**kwargs): # TODO
        return "", True 

class STCufMemcpy(nodes.STNode):

    def __init__(self, *args, **kwargs):
        nodes.STNode.__init__(self, *args, **kwargs)
        self._parse_result = translator.tree.grammar.memcpy.parseString(
            self.statements()[0])[0]

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]): 
        # TODO backend specific, move to cuf subpackage
        # TODO remove completely and make subcase of assignment
        # TODO set index vars from outside
        if "cuf" in opts.source_dialects and isinstance(self.parent, nodes.STContainerBase):
            indent = self.first_line_indent()
            def repl_memcpy_(parse_result):
                dest_name = parse_result.dest_name_fstr()
                src_name = parse_result.src_name_fstr()
                try:
                    dest_indexed_var = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                      dest_name)
                    src_indexed_var  = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                      src_name)
                    dest_on_device = "device" in dest_indexed_var["attributes"]
                    src_on_device  = "device" in src_indexed_var["attributes"] 
                except util.error.LookupError:
                    dest_on_device = False 
                    src_on_device  = False
                if dest_on_device or src_on_device:
                    subst = parse_result.hip_fstr(dest_on_device, src_on_device)
                    return (textwrap.indent(subst,indent), True)
                else:
                    return ("", False) # no transformation; will not be considered
            return repl_memcpy_(self._parse_result)
        else:
            return ("", False)