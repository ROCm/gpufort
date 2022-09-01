# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import grammar
from gpufort import translator

from .. import kernelgen
from .. import opts
from . import render

class HipKernelGeneratorBase(kernelgen.KernelGeneratorBase):
    """Base class for constructing HIP kernel generators."""

    def __init__(self, scope, **kwargs):
        r"""Constructor. 
        :param dict scope: A dictionary containing index entries for the scope that the kernel is in
        :param \*\*kwargs: See below.
        
        :Keyword Arguments:

        * *kernel_name* (`str`):
            Name to give the kernel.
        * *kernel_hash* (`str`):
            Hash code encoding the significant kernel content (expressions and directives).
        * *fortran_snippet* (`str`):
           Original Fortran snippet to put into the kernel documentation. 
           Error handling mode. TODO review
        :param  
        """
        util.kwargs.set_from_kwargs(self, "kernel_name", "mykernel", **kwargs)
        util.kwargs.set_from_kwargs(self, "kernel_hash", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "fortran_snippet", "", **kwargs)
        self.scope = scope
        # to be set by subclass:
        self.c_body = ""
        #
        self.kernel = None

    def get_kernel_args(self):
        """:return: index records for the variables
                    that must be passed as kernel arguments.
        :note: Shared_vars and local_vars must be passed as kernel argument
               as well if the respective variable is an array. 
        """
        return translator.analysis.get_kernel_args(self.kernel["global_vars"],
                                                   self.kernel["global_reduced_vars"],
                                                   self.kernel["shared_vars"],
                                                   self.kernel["local_vars"])

    def _create_kernel_context(self,launch_bounds=None):
        kernel = self._create_kernel_base_context(self.kernel_name,
                                                  self.c_body)
        # '\' is required; cannot be omitted
        kernel["attribute"] = "__global__"
        kernel["return_type"] = "void"
        kernel["launch_bounds"] =  "" if launch_bounds == None else launch_bounds
        kernel["global_vars"] = []
        kernel["global_reduced_vars"] = []
        kernel["shared_vars"] = []
        kernel["local_vars"] = []
        kernel["shared_and_local_array_vars"] = []
        kernel["namespace"] = "_"+self.scope["tag"].replace(":","_")
        return kernel

    def __create_launcher_base_context(self,
                                       kind,
                                       debug_code,
                                       used_modules=[]):
        """Create base context for kernel launcher code generation.
        :param str kind:one of 'hip', 'hip_ps', or 'cpu'.
        """
        launcher_name_tokens = ["launch", self.kernel_name, kind]
        return {
            "kind": kind,
            "name": "_".join(launcher_name_tokens),
            "used_modules": used_modules,
            "debug_code": debug_code,
        }

    def _create_shared_and_local_array_vars(self):
        self.kernel["shared_and_local_array_vars"] = [ivar for ivar in (self.kernel["shared_vars"]+self.kernel["local_vars"]) 
                                                      if ivar["rank"] > 0]

    def create_launcher_context(self, kind, debug_code, used_modules=[]):
        """Create context for HIP kernel launcher code generation.
        :param str kind:one of 'hip', 'hip_ps', or 'cpu'.
        """
        launcher = self.__create_launcher_base_context(kind, debug_code,
                                                       used_modules)
        return launcher
    
    def render_gpu_kernel_prototype_cpp(self):
        return [
            render.render_hip_device_routine_prototype_cpp(self.kernel),
        ]

    def render_gpu_kernel_cpp(self):
        return [
            render.render_hip_device_routine_comment_cpp(self.fortran_snippet),
            render.render_hip_device_routine_cpp(self.kernel),
        ]

    def render_begin_kernel_comment_cpp(self):
        return [" ".join(["//", "BEGIN", self.kernel_name, self.kernel_hash])]

    def render_end_kernel_comment_cpp(self):
        return [" ".join(["//", "END", self.kernel_name, self.kernel_hash])]

    def render_begin_kernel_comment_f03(self):
        return [" ".join(["!", "BEGIN", self.kernel_name, self.kernel_hash])]

    def render_end_kernel_comment_f03(self):
        return [" ".join(["!", "END", self.kernel_name, self.kernel_hash])]

    def render_gpu_launcher_cpp(self, launcher):
        return [render.render_hip_launcher_cpp(self.kernel, launcher)]

    def render_cpu_launcher_cpp(self, launcher):
        return [
            render.render_cpu_routine_cpp(self.kernel),
            render.render_cpu_launcher_cpp(self.kernel, launcher),
        ]

    def render_launcher_interface_f03(self, launcher):
        return [render.render_launcher_f03(self.kernel, launcher)]

    def render_cpu_routine_f03(self, launcher):
        return [
            render.render_cpu_routine_f03(self.kernel, launcher,
                                          self.fortran_snippet)
        ]


# derived classes
class HipKernelGenerator4LoopNest(HipKernelGeneratorBase):

    def __init__(self, ttcomputeconstruct, scope, **kwargs):
        HipKernelGeneratorBase.__init__(self, scope, **kwargs)
        # TODO tensors must be flagged before before lookup index entries
        # translate_compute_construct_to_hip_kernel_body does this (side effect)
        self.c_body, self.problem_size, self.block_size, loop_vars, c_names, c_ranks = \
            translator.codegen.translate_compute_construct_to_hip_kernel_body(ttcomputeconstruct,scope,**kwargs)
        self.kernel = self._create_kernel_context()
       
        self.kernel["global_vars"],\
        self.kernel["global_reduced_vars"],\
        self.kernel["shared_vars"],\
        self.kernel["local_vars"] = translator.analysis.lookup_index_entries_for_vars_in_compute_construct(self.scope,
                                                                                       ttcomputeconstruct,
                                                                                       loop_vars,
                                                                                       c_names,
                                                                                       c_ranks)
        #self._create_shared_and_local_array_vars()

class HipKernelGenerator4CufKernel(HipKernelGeneratorBase):

    def __init__(self, ttprocedurebody, iprocedure, scope, **kwargs):
        HipKernelGeneratorBase.__init__(self, scope, **kwargs)
        #
        self.c_body = translator.codegen.translate_procedure_body_to_hip_kernel_body(ttprocedurebody,scope,**kwargs)
        self.kernel = self._create_kernel_context()
        
        self.kernel["global_vars"],\
        self.kernel["global_reduced_vars"],\
        self.kernel["shared_vars"],\
        self.kernel["local_vars"] = translator.analysis.lookup_index_entries_for_vars_in_procedure_body(self.scope,
                                                                                                        ttprocedurebody,
                                                                                                        iprocedure,
                                                                                                        )
        #self._create_shared_and_local_array_vars()

    def render_cpu_routine_f03(self, launcher):
        return []

    def render_cpu_launcher_cpp(self):
        return []

class HipKernelGenerator4AcceleratorRoutine(HipKernelGenerator4CufKernel):
    def __init__(self, *args, **kwargs):
        HipKernelGenerator4CufKernel.__init__(self, *args, **kwargs)
        return_type, _ = util.kwargs.get_value("return_type", "void", **kwargs)
        self.kernel["return_type"] = return_type
        self.kernel["attribute"] = "__device__"

    def render_gpu_launcher_cpp(self, launcher):
        return []