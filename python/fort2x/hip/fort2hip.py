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

import fort2x.fort2x
import fort2x.hip.derivedtypegen
import fort2x.hip.kernelgen

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"fort2hip_options.py.in")).read())
    
class HipCodeGenerator(fort2x.fort2x.CodeGenerator):
    """Code generator for generating HIP C++ kernels and kernel launchers.
    :param bool emit_fortran_interfaces: Render explicit Fortran interfaces for the extracted kernels.
    :param bool emit_cpu_launcher: Render a launcher that runs the original loop nest on the CPU (only for loop nests).
    :param bool emit_debug_code: Write debug code such as printing of kernel arguments or device array norms and elements
                                   into the generated kernel launchers.
    """
    def __init__(self,
                 stree,
                 index,
                 kernels_to_convert        = ["*"],
                 emit_fortran_interfaces   = True, 
                 emit_debug_code           = False,
                 emit_cpu_launcher         = False,
                 fortran_module_preamble   = FORTAN_MODULE_PREAMBLE,
                 cpp_file_preamble         = CPP_FILE_PREAMBLE,
                 fortran_module_suffix     = FORTRAN_MODULE_SUFFIX,
                 fortran_module_file_ext   = FORTRAN_MODULE_FILE_EXT,
                 cpp_file_ext              = CPP_FILE_EXT):
        fort2x.fort2x.CodeGenerator.__init__(stree,
                                             index,
                                             kernels_to_convert,
                                             fortran_module_preamble,
                                             cpp_file_preamble,
                                             fortran_module_suffix,
                                             fortran_module_file_ext,
                                             cpp_file_ext)
        self.emit_debug_code         = emit_debug_code
        self.emit_cpu_launcher       = emit_cpu_launcher
        self.emit_fortran_interfaces = emit_fortran_interfaces
     
    def __render_kernel(self,
                        kernelgen,
                        cpp_filegen,
                        fortran_filegen,
                        is_loopnest=False):
        cpp_filegen.rendered_kernels += (kernelgen.render_begin_kernel_comment_cpp()
                                        + kernelgen.render_gpu_kernel_cpp()
                                        + kernelgen.render_end_kernel_comment_cpp())
        hip_launcher = kernelgen.create_launcher_context("hip",
                                                         self.emit_debug_code,
                                                         self._make_module_dicts(self.default_modules))
        rendered_launchers_cpp  = kernelgen.render_gpu_launcher_cpp(hip_launcher)
        rendered_interfaces_f03 = kernelgen.render_launcher_interface_f03(hip_launcher)
        
        hip_ps_launcher = copy.deepcopy(hip_launcher)
        hip_ps_launcher["kind"] = "hip_ps"
        
        rendered_launchers_cpp  += kernelgen.render_gpu_launcher_cpp(hip_ps_launcher)
        rendered_interfaces_f03 += kernelgen.render_launcher_interface_f03(hip_ps_launcher)

        if is_loopnest and self.emit_cpu_launcher:
            cpu_launcher = copy.deepcopy(hip_launcher)
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
        
        kernelgen = fort2x.hip.kernelgen_hip.HipKernelGenerator4LoopNest(stloopnest.kernel_name(),
                                                                         stloopnest.kernel_hash(),
                                                                         stloopnest.parse_result,
                                                                         scope,
                                                                         "".join(stloopnest.lines()))

        self.__render_kernel(kernelgen,
                             cpp_filegen,
                             fortran_filegen,
                             is_loopnest=True)

        stloopnest.set_kernel_arguments(kernelgen.get_kernel_arguments)
 
        utils.logging.log_leave_function(LOG_PREFIX,"self._render_loop_nest")
    
    def _render_device_procedure(stprocedure,
                                 iprocedure,
                                 cpp_filegen,
                                 fortran_filegen):
        kernel_name = iprocedure["name"]
        scope = scoper.create_scope(index,
                                    stprocedure._parent.tag())
        if stprocedure.is_kernel_subroutine():  
            kernelgen = fort2x.hip.kernelgen_hip.HipKernelGenerator4CufKernel(iprocedure["name"],
                                                                             stprocedure.kernel_hash(),
                                                                             stprocedure.parse_result,
                                                                             iprocedure,
                                                                             scope,
                                                                             "".join(stlprocedure.lines()))
        else:
            kernelgen = fort2x.hip.kernelgen_hip.HipKernelGenerator4AcceleratorRoutine(stprocedure.kernel_name(),
                                                                                       stprocedure.kernel_hash(),
                                                                                       stprocedure.parse_result,
                                                                                       iprocedure,
                                                                                       scope,
                                                                                       "\n".join(fortran_statements))

        self.__render_kernel(kernelgen,
                             cpp_filegen,
                             fortran_filegen,
                             is_loopnest=False)

        stloopnest.set_kernel_arguments(kernelgen.get_kernel_arguments)
 
        utils.logging.log_leave_function(LOG_PREFIX,"self._render_loop_nest")
    def _render_derived_types(self,
                              itypes,
                              used_modules,
                              cpp_filegen,
                              fortran_modulegen):
        derivedtypegen = DerivedTypeGenerator(itypes,used_modules) 
        cpp_filegen.rendered_types          += derivedtypegen.render_derived_type_definitions_cpp()
        fortran_modulegen.rendered_types    += derivedtypegen.render_derived_type_definitions_f03()
        fortran_modulegen.rendered_routines += derivedtypegen.render_derived_type_routines_f03()
