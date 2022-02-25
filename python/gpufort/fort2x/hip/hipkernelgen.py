# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import kernelgen
from . import render

from gpufort import util
from gpufort import grammar

from .. import opts

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
        * *error_handling* (`str`):
           Error handling mode. TODO review
        :param  
        """
        util.kwargs.set_from_kwargs(self, "kernel_name", "mykernel", **kwargs)
        util.kwargs.set_from_kwargs(self, "kernel_hash", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "fortran_snippet", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "error_handling", None, **kwargs)
        self.scope = scope
        # to be set by subclass:
        self.c_body = ""
        self.all_vars = []
        self.global_reductions = {}
        self.shared_vars = []
        self.local_vars = []
        #
        self.kernel = None

    def get_kernel_arguments(self):
        """:return: index records for the variables
                    that must be passed as kernel arguments.
        """
        return (self.kernel["global_vars"]
                + self.kernel["global_reduced_vars"])

    def _create_kernel_context(self,launch_bounds=None):
        kernel = self._create_kernel_base_context(self.kernel_name,
                                                  self.c_body)
        # '\' is required; cannot be omitted
        kernel["attribute"] = "__global__"
        kernel["return_type"] = "void"
        kernel["launch_bounds"] =  "" if launch_bounds == None else launch_bounds

        kernel["global_vars"],\
        kernel["global_reduced_vars"],\
        kernel["shared_vars"],\
        kernel["local_vars"] = kernelgen.KernelGeneratorBase.lookup_index_entries_for_vars_in_kernel_body(self.scope,
                                                                                                          self.all_vars,
                                                                                                          self.global_reductions,
                                                                                                          self.shared_vars,
                                                                                                          self.local_vars,
                                                                                                          self.error_handling)
        return kernel

    def __create_launcher_base_context(self,
                                       kind,
                                       debug_output,
                                       used_modules=[]):
        """Create base context for kernel launcher code generation.
        :param str kind:one of 'hip', 'hip_ps', or 'cpu'.
        """
        launcher_name_tokens = ["launch", self.kernel_name, kind]
        return {
            "kind": kind,
            "name": "_".join(launcher_name_tokens),
            "used_modules": used_modules,
            "debug_output": debug_output,
        }

    def create_launcher_context(self, kind, debug_output, used_modules=[]):
        """Create context for HIP kernel launcher code generation.
        :param str kind:one of 'hip', 'hip_ps', or 'cpu'.
        """
        launcher = self.__create_launcher_base_context(kind, debug_output,
                                                       used_modules)
        return launcher

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

    def __init__(self, ttloopnest, scope, **kwargs):
        HipKernelGeneratorBase.__init__(self, scope, **kwargs)
        #
        self.all_vars = ttloopnest.vars_in_body()
        self.global_reductions = ttloopnest.gang_team_reductions()
        self.shared_vars = ttloopnest.gang_team_shared_vars()
        self.local_vars = ttloopnest.local_scalars()
        self.c_body = ttloopnest.c_str()
        #
        self.kernel = self._create_kernel_context()
        
        util.logging.log_debug2(opts.log_prefix+".HipKernelGenerator4CufKernel","__init__","".join(
            [
              "all_vars=",str(self.all_vars),"global_reductions=",str(self.global_reductions),
              ",","all_vars=",str(self.shared_vars),",","local_vars=",str(self.local_vars),
            ]))

class HipKernelGenerator4CufKernel(HipKernelGeneratorBase):

    def __init__(self, ttprocedure, iprocedure, scope, **kwargs):
        HipKernelGeneratorBase.__init__(self, scope, **kwargs)
        #
        self.shared_vars = [
            ivar["name"]
            for ivar in iprocedure["variables"]
            if "shared" in ivar["qualifiers"]
        ]
        self.local_vars = [
            ivar["name"]
            for ivar in iprocedure["variables"]
            if ivar["name"] not in iprocedure["dummy_args"]
        ]
        all_var_exprs = kernelgen.KernelGeneratorBase.strip_member_access(ttprocedure.vars_in_body(
        )) # in the body, there might be variables present from used modules
        self.all_vars = iprocedure["dummy_args"] + [
            v for v in all_var_exprs if (v not in iprocedure["dummy_args"] and
                                         v not in grammar.DEVICE_PREDEFINED_VARIABLES)
        ]
        self.global_reductions = {}
        self.c_body = ttprocedure.c_str()
        #
        self.kernel = self._create_kernel_context()

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
