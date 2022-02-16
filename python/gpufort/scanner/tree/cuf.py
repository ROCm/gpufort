# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from . import base
from gpufort import util
from .. import opts
from . import grammar

class CufBackendBase:
    def __init__(self,stnode):
        self._stnode = stnode

CUF_LOOP_KERNEL_BACKENDS = {} 

def register_cuf_backend(name,loop_kernel_generator_class,runtime_module_name):
    if not name in base.SUPPORTED_DESTINATION_DIALECTS:
        base.SUPPORTED_DESTINATION_DIALECTS.append(name)
    base.RUNTIME_MODULE_NAMES[name]     = runtime_module_name
    CUF_LOOP_KERNEL_BACKENDS[name] = loop_kernel_generator_class

class STCufDirective(base.STDirective):
    """This class has the functionality of a kernel if the stored lines contain
    a cuf kernel directivity. 
    """
    def __init__(self,lineno,lines,directive_no):
        base.STDirective.__init__(self,lineno,lines,directive_no,sentinel="!$cuf")
        self._default_present_vars = []
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[]):
        assert False, "Currently, there are only CUF parallel directives"

class STCufLoopNest(STCufDirective,base.STLoopNest):
    def __init__(self,lineno,lines,directive_no):
        STCufDirective.__init__(self,lineno,lines,directive_no)
        base.STLoopNest.__init__(self,lineno,lines)
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[],destination_dialect=""):
        """
        :param destination_dialect: allows to override default if this kernel
                                   should be translated via another backend.
        """
        checked_dialect = base.check_destination_dialect(\
            opts.destination_dialect if not len(destination_dialect) else destination_dialect)
        return CUF_LOOP_KERNEL_BACKENDS[checked_dialect](self).transform(\
          joined_lines,joined_statements,statements_fully_cover_lines,index)

def handle_allocate_cuf(stallocate,joined_statements,index):
    indent = stallocate.first_line_indent()
    # CUF
    transformed       = False
    bytes_per_element = []
    array_qualifiers  = []
    for array_name in stallocate.parse_result.variable_names():
        ivar,_  = scope.search_index_for_variable(index,stallocate.parent.tag(),\
          array_name)
        bytes_per_element.append(ivar["bytes_per_element"])
        qualifier, transformation_required = base.pinned_or_on_device(ivar)
        transformed |= transformation_required
        array_qualifiers.append(qualifier)
    subst = stallocate.parse_result.hip_f_str(bytes_per_element,array_qualifiers,indent=indent).lstrip(" ")
    return (subst, transformed)

def handle_deallocate_cuf(stdeallocate,joined_statements,index):
    indent = stdeallocate.first_line_indent()
    transformed      = False
    array_qualifiers = []
    for array_name in stdeallocate.parse_result.variable_names():
        ivar,_  = scope.search_index_for_variable(index,stdeallocate.parent.tag(),\
          array_name)
        on_device   = base.index_variable_is_on_device(ivar)
        qualifier, transformed1 = base.pinned_or_on_device(ivar)
        transformed |= transformed1
        array_qualifiers.append(qualifier)
    subst = stdeallocate.parse_result.hip_f_str(array_qualifiers,indent=indent).lstrip(" ")
    return (subst, transformed)

@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_cuf(stree,index,destination_dialect=""):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    
    # cublas_v1 detection
    if opts.cublas_version == 1:
        def has_cublas_call_(child):
            return type(child) is base.STCufLibCall and child.has_cublas()
        cuf_cublas_calls = stree.find_all(filter=has_cublas_call_, recursively=True)
        for call in cuf_cublas_calls:
            last_decl_list_node = call.parent.last_entry_in_decl_list() 
            indent = self.first_line_indent()
            last_decl_list_node.add_to_epilog("{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(indent))
 
            local_cublas_calls = call.parent.find_all(filter=has_cublas_call, recursively=False)
            first  = local_cublas_calls[0]
            indent = self.first_line_indent()
            first.add_to_prolog("{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last   = local_cublas_calls[-1]
            indent = self.first_line_indent()
            last.add_to_epilog("{0}hipblasDestroy(hipblasHandle)\n".format(indent))
