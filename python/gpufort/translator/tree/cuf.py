# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from . import base
from . import traversals
from . import fortran
from . import grammar
from . import directives

class TTCufKernelCall(base.TTNode):

    def _assign_fields(self, tokens):

        self._kernel_name = tokens[0]
        self._grid = tokens[1][0]
        self._block = tokens[1][1]
        self._sharedmem = tokens[1][2]
        self._stream = tokens[1][3]
        self._args = tokens[2] 

    def kernel_name_fstr(self):
        return traversals.make_fstr(self._kernel_name)
    
    def _postprocess_dim3(dim3):
        try:
            int_val = int(dim3)
            is_one = int_val == 1
        except:
            is_one = False
        if is_one:
            return "dim3Ones" # specifically for Eigensolver_gpu
        else:
            return dim3

    def grid_fstr(self):
        return traversals.make_fstr(
          self._postprocess_dim3(self._grid)
        )

    def block_fstr(self):
        return traversals.make_fstr(
          self._postprocess_dim3(self._block)
        )

    def use_default_stream(self):
        try:
            return int(self._stream) < 1
        except:
            return False

    def stream_fstr(self):
        if self.use_default_stream():
            return "c_null_ptr"
        else:
            return traversals.make_fstr(self._stream)

    def sharedmem_fstr(self):
        return traversals.make_fstr(self._sharedmem)

# todo: keep for reference, a while
#class CufDeviceSpec(directives.IDeviceSpec):
#    def __init__(self,tokens):
#        self._num_outer_loops_to_map = int(tokens[0])
#        self._grid = tokens[1][0]
#        self._block = tokens[1][1]
#    
#    def parallelism(self):
#        return Parallelism.GANG_WORKER_VECTOR
#
#    def order_of_iterates(self):
#        return IterateOrder.INDEPENDENT

class TTCufKernelDoArgGrid(base.TTNode):
    def _assign_fields(self, tokens):
        self.expr = tokens[0]
class TTCufKernelDoArgBlock(base.TTNode):
    def _assign_fields(self, tokens):
        self.expr = tokens[0]
class TTCufKernelDoArgNumLoops(base.TTNode):
    def _assign_fields(self, tokens):
        self.expr = tokens[0]
class TTCufKernelDoArgSharedmem(base.TTNode):
    def _assign_fields(self, tokens):
        self.expr = tokens[0]
class TTCufKernelDoArgStream(base.TTNode):
    def _assign_fields(self, tokens):
        self.expr = tokens[0]

class TTCufKernelDo(base.TTNode):

    def _assign_fields(self, tokens):
        self._num_outer_loops_to_map = 1
        self._grid = None 
        self._block = None
        self._sharedmem = None
        self._stream = None
        if len(tokens) > 0:
            self._num_outer_loops_to_map = tokens[0]
            launch_params = tokens[1]
            if launch_params != None:
                self._grid = launch_params[0] 
                self._block = launch_params[1]
                if len(launch_params) > 2:
                    if len(launch_params) == 4:
                        self._sharedmem = launch_params[2] # optional
                    self._stream = launch_params[-1] # optional

        def child_nodes(self):
            return [self._num_outer_loops_to_map,
                    self._grid,
                    self._block,
                    self._sharedmem,
                    self._stream]

#    def discover_reduction_candidates(self):
#        return True
#
#    def all_arrays_are_on_device(self):
#        return True
#    def stream(self, converter=traversals.make_fstr):
#        return converter(self._stream)
#
#    def sharedmem(self, converter=traversals.make_fstr):
#        return converter(self._sharedmem)
#
#    def use_default_stream(self):
#        try:
#            return int(self._stream) < 1
#        except:
#            return False
#    
#    def num_collapse(self):
#        return self._num_outer_loops_to_map
#
#    def num_gangs(self, converter=traversals.make_fstr):
#        if self._grid == "*":
#            return [grammar.CLAUSE_NOT_FOUND] * self._num_outer_loops_to_map
#        elif isinstance(self._block, fortran.TTValue):
#            # todo: Check if TTValue is actually a dim3 or not
#            result = []
#            for i in range(0, self._num_outer_loops_to_map):
#                result.append(converter(self._grid) + "%" + chr(ord('x') + i))
#            return result
#        else:
#            return [converter(gridDim) for gridDim in self._grid]
#
#    def num_threads_in_block(self, converter=traversals.make_fstr):
#        if self._block == "*":
#            return [grammar.CLAUSE_NOT_FOUND] * self._num_outer_loops_to_map
#        elif isinstance(self._block, fortran.TTValue):
#            # todo: Check if TTValue is actually a dim3 or not
#            result = []
#            for i in range(0, self._num_outer_loops_to_map):
#                result.append(converter(self._block) + "%" + chr(ord('x') + i))
#            return result
#        else:
#            return [converter(blockDim) for blockDim in self._block]
#        
#    def grid_expr_fstr(self):
#        """ only CUF """
#        return traversals.make_fstr(self._grid)
#
#    def block_expr_fstr(self):
#        """ only CUF """
#        return traversals.make_fstr(self._block)
#
#    def num_dimensions(self):
#        """
#        Get the number of grid and block dimensions.
#        This might differ from the number of nested loops.
#        """
#        if opts.loop_collapse_strategy == "grid":
#            return int(self._device_spec._num_outer_loops_to_map)
#        else:
#            return 1
#
#    def present_by_default(self):
#        return True
#
#    def cstr(self):
#        result = "// NOTE: The following information was given in the orignal CUDA Fortran kernel pragma:\n"
#        result += "// - Nested outer-most do-loops that are directly mapped to threads: {}\n".format(
#            traversals.make_cstr(self._num_outer_loops_to_map))
#        result += "// - Number of blocks: {}. ('-1' means not specified)\n".format(
#            traversals.make_cstr(self._grid))
#        result += "// - Threads per block: {}. ('-1' means not specified)\n".format(
#            traversals.make_cstr(self._block))
#        result += "// - Shared Memory: {}\n".format(
#            traversals.make_fstr(self._sharedmem))
#        result += "// - Stream: {}\n".format(traversals.make_fstr(self._stream))
#        return result

    def omp_fstr(self,
                  arrays_in_body=set(),
                  inout_arrays_in_body=set(),
                  reduction={},
                  depend={},
                  loop_type="do"):
        result = "!$omp target teams distribute parallel " + loop_type
        grid = self._device_spec.num_gangs()
        block = self._device_spec.num_threads_in_block()
        num_teams = ""
        thread_limit = ""
        first = True
        for val in grid:
            if val != grammar.CLAUSE_NOT_FOUND:
                num_teams = "*" + val if not first else val
                first = False
        first = True
        for val in block:
            if val != grammar.CLAUSE_NOT_FOUND:
                thread_limit = "*" + val if not first else val
                first = False
        if len(num_teams):
            result += " num_teams(" + num_teams + ")"
        if len(thread_limit):
            result += " thread_limit(" + thread_limit + ")"
        # reduction vars
        for kind, variables in reduction.items():
            if len(variables):
                result += " reduction({0}:{1})".format(kind,
                                                       ",".join(variables))
        # if, async
        if self.stream() != str(grammar.CLAUSE_NOT_FOUND):
            result += " nowait"
            if len(depend):
                for kind, variables in depend.items():
                    result += " depend(" + kind + ":" + ",".join(
                        variables) + ")"
            else:
                in_arrays_in_body = [
                    el for el in arrays_in_body
                    if el not in inout_arrays_in_body
                ]
                if len(in_arrays_in_body):
                    result += " depend(" + kind + ":" + ",".join(
                        in_arrays_in_body) + ")"
                if len(inout_arrays_in_body):
                    result += " depend(" + kind + ":" + ",".join(
                        inout_arrays_in_body) + ")"
        return result

class TTCufAllocated(base.TTNode):
    """
     
    For `type(c_ptr)` variables the `allocated` check
    does not work. We need to use `c_associated` instead.
    
    For Fortran pointer variables, we need the `associated(<var>)`
    intrinsic to check if they are associated with any memory.
    """

    def _assign_fields(self, tokens):
        self._var = tokens[0]

    def var_name(self,converter=traversals.make_fstr):
        return self._var.identifier_part(converter)

    def fstr(self, var_is_c_ptr=False):
        if var_is_c_ptr:
            return "c_associated({0})".format(traversals.make_fstr(self._var))
        else:
            return "associated({0})".format(traversals.make_fstr(self._var))


class TTCufNonZeroCheck(base.TTNode):
    """
    For `type(c_ptr)` variables that replace CUDA Fortran stream and , the standard non-zero check
    does not work. We need to replace this by `c_associated(var)`.
    """

    def _assign_fields(self, tokens):
        self._lhs = tokens

    def lhs_fstr(self):
        return traversals.make_fstr(self._lhs)

    def fstr(self, lhs_is_c_ptr=False):
        if lhs_is_c_ptr:
            return "c_associated({0})".format(traversals.make_fstr(self._lhs))
        else:
            return "associated({0})".format(traversals.make_fstr(self._lhs))


class TTCufPointerAssignment(base.TTNode):
    """
    For `type(c_ptr)` variables that replace device array pointers,
    the Fortran pointer assignment operator `=>` must be replaced by `=`.
 
    Calling function needs to check if at least one (should be both in any case)
    pointer is pointing to data on the device. 
    """

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def lhs_fstr(self):
        return traversals.make_fstr(self._lhs)

    def rhs_fstr(self):
        return traversals.make_fstr(self._rhs)

    def fstr(self, vars_are_c_ptrs=False, lhs_bound_var_names=[]):
        if vars_are_c_ptrs:
            lhs_name = self.lhs_fstr()
            rhs_name = self.rhs_fstr()
            bound_var_assignments = "\n".join([
                "{0} = {1}".format(el, el.replace(lhs_name, rhs_name))
                for el in lhs_bound_var_names
            ])
            return "{0} = {1}\n{2}".format(lhs_name, rhs_name,
                                           bound_var_assignments)
        else:
            return "{0} => {1}\n{2}".format(lhs_name, rhs_name)


class CufMemcpyBase():
    """
    Abstract base class.
    Subclasses initialize members (api,dest,src)
    """

    def hip_ap_i(self):
        return "hip" + self._api[4:].title().replace("async", "Async")

    def dest_fstr(self, on_device, bytes_per_element):
        """
        :return: simply returns the Fortran pointer representation of the destination. 
        """
        return traversals.make_fstr(self._dest)

    def src_fstr(self, on_device, bytes_per_element):
        """
        :return: the Fortran pointer representation of the source.
        """
        return traversals.make_fstr(self._src)

    def dest_name_fstr(self):
        """
        :return: name of destination variable; may contain '%' if derived type member.
        """
        return traversals.make_fstr(self._dest.identifier_part(traversals.make_fstr))

    def src_name_fstr(self):
        """
        :return: name of source variable; may contain '%' if derived type member.
        """
        return traversals.make_fstr(self._src.identifier_part(traversals.make_fstr))

    def dest_has_args(self):
        return self._dest.args() != None

    def src_has_args(self):
        return self._src.args() != None

    def size_fstr(self, name, bytes_per_element=1):
        """
        The size of transferred memory (in bytes if bytes per element are given).
        multiplication by 1_8 ensures this is a type
        compatible with `integer(c_size_t)`.
        """
        assert name in self.__dict__
        size = traversals.make_fstr(self.__dict__[name])
        if bytes_per_element != 1:
            return "1_8 * ({0}) * ({1})".format(size, bytes_per_element)
        else:
            return size

    def memcpy_kind(self, dest_on_device, src_on_device):
        if self._memcpy_kind != None:
            return self._memcpy_kind
        else:
            result = "hipMemcpy"
            result += "Device" if src_on_device else "Host"
            result += "ToDevice" if dest_on_device else "ToHost"
            return result

    def hip_stream_fstr(self):
        if str(self._stream) == "0":
            return "c_null_ptr"
        else:
            return traversals.make_fstr(self._stream)


class TTCufMemcpyIntrinsic(base.TTNode, CufMemcpyBase):
    # dest,src,count,[,stream] # kind is inferred from dest and src
    def _assign_fields(self, tokens):
        self._dest = tokens[0]
        self._src = tokens[1]
        self._memcpy_kind = None

    def hip_fstr(self,
                  dest_on_device,
                  src_on_device,
                  bytes_per_element=1): # todo: backend specific
        api = "hipMemcpy"
        args = []
        args.append(self.dest_fstr(dest_on_device, bytes_per_element))
        args.append(self.src_fstr(src_on_device, bytes_per_element))
        args.append(self.memcpy_kind(dest_on_device, src_on_device))
        return "call hipCheck({api}({args}))".format(
            api=api, args=", ".join(args))


class TTCufCudaMemcpy(base.TTNode, CufMemcpyBase):
    # dest,src,count,[,stream] # kind is inferred from dest and src
    def _assign_fields(self, tokens):
        #print(tokens)
        self._api = tokens[0]
        self._dest = tokens[1]
        self._src = tokens[2]
        self._count = tokens[3]
        self._memcpy_kind = tokens[4]
        self._stream = tokens[5]

    def hip_fstr(self, dest_on_device, src_on_device, bytes_per_element=1):
        api = self.hip_ap_i()
        args = []
        args.append(self.dest_fstr(dest_on_device, bytes_per_element))
        args.append(self.src_fstr(src_on_device, bytes_per_element))
        args.append(self.size_fstr("_count", bytes_per_element))
        args.append(self.memcpy_kind(dest_on_device, src_on_device))
        if "Async" in api:
            args.append(traversals.make_fstr(self._stream))
        return "{api}({args})".format(api=api, args=",".join(args))


class TTCufCudaMemcpy2D(base.TTNode, CufMemcpyBase):
    # dest,dpitch(count),src,spitch(count),width(count),height(count)[,stream] # kind is inferred from dest and src
    def _assign_fields(self, tokens):
        self._api = tokens[0]
        self._dest = tokens[1]
        self._dpitch = tokens[2]
        self._src = tokens[3]
        self._spitch = tokens[4]
        self._width = tokens[5]
        self._height = tokens[6]
        self._memcpy_kind = tokens[7]
        self._stream = tokens[8]

    def hip_fstr(self, dest_on_device, src_on_device, bytes_per_element=1):
        api = self.hip_ap_i()
        args = []
        args.append(self.dest_fstr(dest_on_device, bytes_per_element))
        args.append(self.size_fstr("_dpitch", bytes_per_element))
        args.append(self.src_fstr(src_on_device, bytes_per_element))
        args.append(self.size_fstr("_spitch", bytes_per_element))
        args.append(self.size_fstr("_width", bytes_per_element))
        args.append(self.size_fstr("_height", 1))
        args.append(self.memcpy_kind(dest_on_device, src_on_device))
        if "Async" in api:
            args.append(self.hip_stream_fstr())
        return "{api}({args})".format(api=api, args=",".join(args))


class TTCufCudaMemcpy3D(base.TTNode, CufMemcpyBase):
    # dest,dpitch(count),src,spitch(count),width(count),height(count),depth(count),[,stream] # kind is inferred from dest and src
    def _assign_fields(self, tokens):
        self._api = tokens[0]
        self._dest = tokens[1]
        self._dpitch = tokens[2]
        self._src = tokens[3]
        self._spitch = tokens[4]
        self._width = tokens[5]
        self._height = tokens[6]
        self._depth = tokens[7]
        self._memcpy_kind = tokens[8]
        self._stream = tokens[9]

    def fstr(self,
              dest_on_device,
              src_on_device,
              bytes_per_element=1,
              indent=""):
        api = self.hip_ap_i()
        args = []
        args.append(self.dest_fstr(dest_on_device, bytes_per_element))
        args.append(self.size_fstr("_dpitch", bytes_per_element))
        args.append(self.src_fstr(src_on_device, bytes_per_element))
        args.append(self.size_fstr("_spitch", bytes_per_element))
        args.append(self.size_fstr("_width", bytes_per_element))
        args.append(self.size_fstr("_height", 1))
        args.append(self.size_fstr("_depth", 1))
        args.append(self.memcpy_kind(dest_on_device, src_on_device))
        if "Async" in api:
            args.append(self.hip_stream_fstr())
        return "{api}({args})".format(api=api, args=",".join(args))


class TTCufCublasCall(base.TTNode):

    def _assign_fields(self, tokens):
        self._api = tokens[0] # does not include cublas
        self._args = tokens[1]

    def hip_fstr(self, indent=""):
        api = "hipblas" + traversals.make_fstr(self._api)
        args = []
        if opts.cublas_version == 1:
            args.append("hipblasHandle")
        else:
            api = api.split("_")[0] # remove _v2 if present
        args += [traversals.make_fstr(arg) for arg in self._args]
        result = "{2}{0}({1})".format(api, ",".join(args), indent)
        cublas_operation_type = Regex("'[NTCntc]'").setParseAction(
            lambda tokens: "HIPBLAS_OP_" + tokens[0].strip("'").upper())
        result = cublas_operation_type.transformString(result)
        return result


## Link actions
grammar.cuf_kernel_do_arg_block.setParseAction(TTCufKernelDoArgBlock)
grammar.cuf_kernel_do_arg_grid.setParseAction(TTCufKernelDoArgGrid)
grammar.cuf_kernel_do_arg_num_loops.setParseAction(TTCufKernelDoArgNumLoops)
grammar.cuf_kernel_do_arg_sharedmem.setParseAction(TTCufKernelDoArgSharedmem)
grammar.cuf_kernel_do_arg_stream.setParseAction(TTCufKernelDoArgStream)
grammar.cuf_kernel_do.setParseAction(TTCufKernelDo)
grammar.allocated.setParseAction(TTCufAllocated)
grammar.memcpy.setParseAction(TTCufMemcpyIntrinsic)
grammar.non_zero_check.setParseAction(TTCufNonZeroCheck)
#pointer_assignment.setParseAction(TTCufPointerAssignment)
grammar.cuf_cudamemcpy.setParseAction(TTCufCudaMemcpy)
grammar.cuf_cudamemcpy2D.setParseAction(TTCufCudaMemcpy2D)
grammar.cuf_cudamemcpy3D.setParseAction(TTCufCudaMemcpy3D)
grammar.cuf_cublas_call.setParseAction(TTCufCublasCall)
