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
    def _render_loop_nest(self,
                          stkernel,
                          cppfilegen,
                          f03filegen):
        utils.logging.log_enter_function(LOG_PREFIX,"self.__render_loop_nest")
    
        ttloopkernel = stkernel.parse_result
        scope        = scoper.create_scope(stkernel._parent.tag())
        
        #print(self.ttloopnest.c_str())
        self.kernelgen = fort2x.hip.kernelgen_hip.HipKernelGenerator4LoopNest(stkernel.kernel_name(),
                                                                              stkernel.kernel_hash(),
                                                                              ttloopnest,
                                                                              scope,
                                                                              "\n".join(fortran_statements))
    
        hip_context["have_reductions"] = False # |= len(reduction_ops)
    
    def _render_device_procedure(stnode,
                                 index,
                                 current_cppfilegen,
                                 current_f03filegen):
        pass
    def _render_kernel_procedure(stnode,
                                 index,
                                 current_cppfilegen,
                                 current_f03filegen):
        pass
    def _render_derived_types(self,
                              itypes,
                              used_modules,
                              cpp_filegen,
                              fortran_modulegen):
        derivedtypegen = DerivedTypeGenerator(itypes,used_modules) 
        cpp_filegen.snippets                += derivedtypegen.render_derived_type_definitions_cpp()
        fortran_modulegen.rendered_types    += derivedtypegen.render_derived_type_definitions_f03()
        fortran_modulegen.rendered_routines += derivedtypegen.render_derived_type_routines_f03()
