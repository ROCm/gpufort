# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import copy
import logging

import addtoplevelpath
import translator.translator as translator
import indexer.indexer as indexer
import indexer.scoper as scoper
import scanner.scanner as scanner
import utils.logging
import utils.fileutils
import utils.kwargs

import fort2x.fort2x
import fort2x.derivedtypegen
import fort2x.hip.kernelgen

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"fort2hip_options.py.in")).read())
    
class HipCodeGenerator(fort2x.fort2x.CodeGenerator):
    """Code generator for generating HIP C++ kernels and kernel launchers
       that can be called from Fortran code."""
    def __init__(self,
                 stree,
                 index,
                 **kwargs):
        r"""Constructor.
        :param stree: Scanner tree created by GPUFORT's scanner component.
        :param index: Index data structure created by GPUFORT's indexer component.
        :param \*\*kwargs: See below.
        
        :Keyword Arguments:

        * *kernels_to_convert* (`list`):
            Filter the kernels to convert to C++ by their name or their id. Pass ['*'] 
            to extract all kernels [default: ['*']]
        * *cpp_file_preamble* (`str`):
            A preamble to write at the top of the files produced by the C++ generators
            that can be created by this class [default: fort2x.CPP_FILE_PREAMBLE].
        * *cpp_file_ext* (`str`):
            File extension for the generated C++ files [default: fort2x.CPP_FILE_EXT].
        * *emit_fortran_interfaces* (``bool``):
            Render **explicit** Fortran interfaces for the extracted kernels.
        * *emit_grid_launcher* (``bool``):
            Only for loop nests: Render a launcher that takes the grid as first argument [default: true].
            This launcher is always rendered for kernel procedures, i.e. this option 
            does not affect them.
        * *emit_problem_size_launcher* (``bool``):
            Only for loop nests: Render a launcher that takes the problem size as first argument [default: true].
        * *emit_cpu_launcher* (``bool``):
            Only for loop nests: Render a launcher that runs the original loop nest on the CPU [default: False].
        * *emit_debug_code* (``bool``):
            Write debug code such as printing of kernel arguments or device array norms and elements
            into the generated kernel launchers.
"""
        fort2x.fort2x.CodeGenerator.__init__(self,stree,index,**kwargs)
        utils.kwargs.set_from_kwargs(self,"emit_debug_code",True,**kwargs)
        utils.kwargs.set_from_kwargs(self,"emit_grid_launcher",False,**kwargs)
        utils.kwargs.set_from_kwargs(self,"emit_problem_size_launcher",True,**kwargs)
        utils.kwargs.set_from_kwargs(self,"emit_cpu_launcher",False,**kwargs)
        utils.kwargs.set_from_kwargs(self,"emit_fortran_interfaces",False,**kwargs)
     
    def __render_kernel(self,
                        kernelgen,
                        cpp_filegen,
                        fortran_filegen,
                        is_loopnest=False):
        cpp_filegen.rendered_kernels += (kernelgen.render_begin_kernel_comment_cpp()
                                        + kernelgen.render_gpu_kernel_cpp()
                                        + kernelgen.render_end_kernel_comment_cpp())
        rendered_launchers_cpp  = []
        rendered_interfaces_f03 = []
        grid_launcher = kernelgen.create_launcher_context("hip",
                                                          self.emit_debug_code,
                                                          fortran_filegen.used_modules)
        if not is_loopnest or self.emit_grid_launcher:
            kernelgen.render_gpu_launcher_cpp(grid_launcher)            
            kernelgen.render_launcher_interface_f03(grid_launcher)

        if is_loopnest and self.emit_problem_size_launcher:
            problem_size_launcher = copy.deepcopy(grid_launcher)
            problem_size_launcher["kind"] = "hip_ps"
            rendered_launchers_cpp  += kernelgen.render_gpu_launcher_cpp(problem_size_launcher)
            rendered_interfaces_f03 += kernelgen.render_launcher_interface_f03(problem_size_launcher)

        if is_loopnest and self.emit_cpu_launcher:
            cpu_launcher = copy.deepcopy(grid_launcher)
            cpu_launcher["kind"] = "cpu"
            rendered_launchers_cpp  += kernelgen.render_cpu_launcher_cpp(cpu_launcher)
            rendered_interfaces_f03 += kernelgen.render_launcher_interface_f03(cpu_launcher)
            fortran_filegen.rendered_routines += (kernelgen.render_begin_kernel_comment_f03()
                                                 + kernelgen.render_cpu_routine_f03(cpu_launcher)
                                                 + kernelgen.render_end_kernel_comment_f03())
                    
        if len(rendered_launchers_cpp):
            cpp_filegen.rendered_launchers += (kernelgen.render_begin_kernel_comment_cpp()
                                              + rendered_launchers_cpp
                                              + kernelgen.render_end_kernel_comment_cpp())
        if len(rendered_interfaces_f03):
            fortran_filegen.rendered_interfaces += (kernelgen.render_begin_kernel_comment_f03()
                                                   + rendered_interfaces_f03
                                                   + kernelgen.render_end_kernel_comment_f03())
    def _render_loop_nest(self,
                          stloopnest,
                          cpp_filegen,
                          fortran_filegen):
        """
        :note: Writes back to stloopnest argument.
        """
        utils.logging.log_enter_function(LOG_PREFIX,"self._render_loop_nest")
    
        scope = scoper.create_scope(self.index,
                                    stloopnest._parent.tag())
        
        kernelgen = fort2x.hip.kernelgen.HipKernelGenerator4LoopNest(stloopnest.kernel_name(),
                                                                         stloopnest.kernel_hash(),
                                                                         stloopnest.parse_result,
                                                                         scope,
                                                                         "".join(stloopnest.lines()))

        self.__render_kernel(kernelgen,
                             cpp_filegen,
                             fortran_filegen,
                             is_loopnest=True)

        # stloopnest.set_kernel_arguments(kernelgen.get_kernel_arguments)
 
        utils.logging.log_leave_function(LOG_PREFIX,"self._render_loop_nest")
    
    def _render_device_procedure(stprocedure,
                                 iprocedure,
                                 cpp_filegen,
                                 fortran_filegen):
        kernel_name = iprocedure["name"]
        scope = scoper.create_scope(index,
                                    stprocedure._parent.tag())
        if stprocedure.is_kernel_subroutine():  
            kernelgen = fort2x.hip.kernelgen.HipKernelGenerator4CufKernel(iprocedure["name"],
                                                                             stprocedure.kernel_hash(),
                                                                             stprocedure.parse_result,
                                                                             iprocedure,
                                                                             scope,
                                                                             "".join(stlprocedure.lines()))
        else:
            kernelgen = fort2x.hip.kernelgen.HipKernelGenerator4AcceleratorRoutine(stprocedure.kernel_name(),
                                                                                       stprocedure.kernel_hash(),
                                                                                       stprocedure.parse_result,
                                                                                       iprocedure,
                                                                                       scope,
                                                                                       "\n".join(fortran_statements))

        self.__render_kernel(kernelgen,
                             cpp_filegen,
                             fortran_filegen,
                             is_loopnest=False)

        # stloopnest.set_kernel_arguments(kernelgen.get_kernel_arguments)
 
        utils.logging.log_leave_function(LOG_PREFIX,"self._render_loop_nest")
    def _render_derived_types(self,
                              itypes,
                              cpp_filegen,
                              fortran_modulegen):
        derivedtypegen = fort2x.derivedtypegen.DerivedTypeGenerator(itypes,[]) 
        cpp_filegen.rendered_types          += derivedtypegen.render_derived_type_definitions_cpp()
        fortran_modulegen.rendered_types    += derivedtypegen.render_derived_type_definitions_f03()
        fortran_modulegen.rendered_routines += derivedtypegen.render_derived_type_routines_f03()
