# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
    def __init__(self, first_linemap, first_linemap_first_statement, directive_no):
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   directive_no,
                                   sentinel="!$cuf")
        self._default_present_vars = []

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        assert False, "Currently, there are only CUF parallel directives"

class STCufLoopNest(STCufDirective, nodes.STLoopNest):
    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, func):
        cls._backends.append((dest_dialects, func))

    def __init__(self, first_linemap, first_linemap_first_statement, directive_no):
        STCufDirective.__init__(self, first_linemap, first_linemap_first_statement, directive_no)
        nodes.STLoopNest.__init__(self, first_linemap, first_linemap_first_statement)
        self.dest_dialect = opts.destination_dialect

    def transform(self,joined_lines,joined_statements,*args,**kwargs):
        result = joined_statements
        transformed = False
        if "cuf" in opts.source_dialects:
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
        if "cuf" in opts.source_dialects:
            old_snippet = snippet
            indent = self.first_line_indent()
            if not opts.keep_cuda_lib_names:

                def repl_memcpy(parse_result):
                    dest_name = parse_result.dest_name_f_str()
                    src_name = parse_result.src_name_f_str()
                    dest_indexed_var = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                      dest_name)
                    src_indexed_var  = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                      src_name)
                    dest_on_device = index_var_is_on_device(dest_indexed_var)
                    src_on_device = index_var_is_on_device(src_indexed_var)
                    subst = parse_result.hip_f_str(dest_on_device, src_on_device)
                    return (subst, True)

                snippet, _ = util.pyparsing.replace_all(
                    snippet, translator.tree.grammar.cuf_cudamemcpy_variant,
                    repl_memcpy)

            def repl_cublas(parse_result):
                subst = parse_result.f_str(indent)
                return (subst, True)

            snippet, have_cublas = util.pyparsing.replace_all(
                snippet, translator.tree.grammar.cuf_cublas_call, repl_cublas)
            if have_cublas:
                self._has_cublas = True
            for elem in CUDA_RUNTIME_ENUMS:
                snippet = replace_ignore_case(
                    elem,
                    elem.replace("cuda", "hip").replace("CUDA", "HIP"), snippet)
            for elem in CUDA_LIB_ENUMS:
                snippet = replace_ignore_case(
                    elem,
                    elem.replace("cu", "hip").replace("CU", "HIP"), snippet)
            for elem in ALL_HOST_ROUTINES: # runtime routines
                snippet = replace_ignore_case(elem, elem.replace("cuda", "hip"),
                                              snippet)
            for elem in CUDA_MATH_LIB_FUNCTIONS:
                snippet = replace_ignore_case(elem, elem.replace("cu", "hip"),
                                              snippet)
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
        if "cuf" in opts.source_dialects:
            snippet = joined_statements
            for tokens, start, end in translator.tree.grammar.cuf_kernel_call.scanString(
                    snippet):
                parse_result = tokens[0]
                kernel_args = []
                for ttexpr in parse_result._args:
                    # expand array arguments
                    max_rank = 0
                    for rvalue in translator.tree.find_all(ttexpr,
                                                           translator.tree.TTRValue):
                        # TODO lookup the procedure first
                        ivar = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                           rvalue.name())
                        max_rank = max(max_rank, ivar["rank"])
                    expr_f_str = translator.tree.make_f_str(ttexpr)
                    if max_rank > 0:
                        kernel_args.append("c_loc(" + expr_f_str + ")")
                    else:
                        kernel_args.append(expr_f_str)
                    for rank in range(1, max_rank + 1):
                        kernel_args.append("size({0},{1})".format(
                            expr_f_str, rank))
                    for rank in range(1, max_rank + 1):
                        kernel_args.append("lbound({0},{1})".format(
                            expr_f_str, rank))
                kernel_launch_info = translator.tree.grammar.cuf_kernel_call.parseString(
                    self.first_statement())[0]
                subst="call launch_{0}({1},{2},{3},{4},{5})".format(\
                  kernel_launch_info.kernel_name_f_str(),\
                  kernel_launch_info.grid_f_str(),
                  kernel_launch_info.block_f_str(),\
                  kernel_launch_info.sharedmem_f_str(),
                  kernel_launch_info.stream_f_str(),\
                  ",".join(kernel_args)
                )
                snippet = snippet.replace(snippet[start:end], subst)
                break
            return snippet, True
        else:
            return "", False

class STCufAttributes(nodes.STNode):
    """CUDA Fortran specific intrinsic that needs to be removed/commented out
    in any case.
    """
    def transform(self,*args,**kwargs): # TODO
        return "", "cuf" in opts.source_dialects
