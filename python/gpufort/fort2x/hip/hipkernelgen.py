# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import kernelgen
from . import render
from gpufort import util

class HipKernelGeneratorBase(kernelgen.KernelGeneratorBase):
    """Base class for constructing HIP kernel generators."""
    def __init__(self,
                 scope,
                 **kwargs):
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
        util.kwargs.set_from_kwargs(self,"kernel_name","mykernel",**kwargs)
        util.kwargs.set_from_kwargs(self,"kernel_hash","",**kwargs)
        util.kwargs.set_from_kwargs(self,"fortran_snippet","",**kwargs)
        util.kwargs.set_from_kwargs(self,"error_handling",None,**kwargs)
        self.scope           = scope
        # to be set by subclass:
        self.c_body            = ""
        self.all_vars          = []
        self.global_reductions = {}
        self.shared_vars       = []
        self.local_vars        = []
        #
        self.kernel = None
    def get_kernel_arguments(self):
        """:return: index records for the variables
                    that must be passed as kernel arguments.
        """
        return (self.kernel["global_vars"]
               + self.kernel["global_reduced_vars"])
    def _create_kernel_context(self):
        kernel = self._create_kernel_base_context(self.kernel_name,
                                                  self.c_body)
        # '\' is required; cannot be omitted
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
        launcher_name_tokens=["launch",self.kernel_name,kind]
        if kind == "hip":
            launcher_name_tokens=["launch",self.kernel_name]
        elif kind == "hip_ps":
            launcher_name_tokens=["launch",self.kernel_name,"ps"]
        else:
            launcher_name_tokens=["launch",self.kernel_name,"cpu"]
        return {
               "kind"         : kind,
               "name"         : "_".join(launcher_name_tokens),
               "used_modules" : used_modules,
               "debug_output" : debug_output,
               }
    def create_launcher_context(self,
                                kind,
                                debug_output,
                                used_modules = []):
        """Create context for HIP kernel launcher code generation.
        :param str kind:one of 'hip', 'hip_ps', or 'cpu'.
        """
        launcher = self.__create_launcher_base_context(kind,
                                                       debug_output,
                                                       used_modules)
        return launcher
    def render_gpu_kernel_cpp(self):
        return [
               render.render_hip_kernel_comment_cpp(self.fortran_snippet),
               render.render_hip_kernel_cpp(self.kernel),
               ]
    def render_begin_kernel_comment_cpp(self):
        return [" ".join(["//","BEGIN",self.kernel_name,self.kernel_hash])]
    def render_end_kernel_comment_cpp(self):
        return [" ".join(["//","END",self.kernel_name,self.kernel_hash])]
    def render_begin_kernel_comment_f03(self):
        return [" ".join(["!","BEGIN",self.kernel_name,self.kernel_hash])]
    def render_end_kernel_comment_f03(self):
        return [" ".join(["!","END",self.kernel_name,self.kernel_hash])]
    def render_gpu_launcher_cpp(self,launcher):
        return [render.render_hip_launcher_cpp(self.kernel,
                                                          launcher)]
    def render_cpu_launcher_cpp(self,launcher):
        return [
               render.render_cpu_routine_cpp(self.kernel),
               render.render_cpu_launcher_cpp(self.kernel,launcher),
               ]
    def render_launcher_interface_f03(self,launcher):
        return [render.render_launcher_f03(self.kernel,launcher)]
    def render_cpu_routine_f03(self,launcher):
        return [render.render_cpu_routine_f03(self.kernel,launcher,self.fortran_snippet)]

# derived classes
class HipKernelGenerator4LoopNest(HipKernelGeneratorBase):
    def __init__(self,
                 ttloopnest,
                 scope,
                 **kwargs):
        HipKernelGeneratorBase.__init__(self,scope,**kwargs)
        #
        self.all_vars          = ttloopnest.vars_in_body()
        self.global_reductions = ttloopnest.gang_team_reductions()
        self.shared_vars       = ttloopnest.gang_team_shared_vars()
        self.local_vars        = ttloopnest.local_scalars()
        self.c_body            = ttloopnest.c_str()
        # 
        self.kernel            = self._create_kernel_context()

class HipKernelGenerator4CufKernel(HipKernelGeneratorBase):
    def __init__(self,
                 ttprocedure,
                 iprocedure,
                 scope,
                 **kwargs):
        HipKernelGeneratorBase.__init__(self,scope,**kwargs)
        #
        self.shared_vars       = [ivar["name"] for ivar in iprocedure["variables"] if "shared" in ivar["qualifiers"]]
        self.local_vars        = [ivar["name"] for ivar in iprocedure["variables"] if ivar["name"] not in iprocedure["dummy_args"]]
        all_var_exprs          = self._strip_member_access(ttprocedure.vars_in_body()) # in the body, there might be variables present from used modules
        self.all_vars          = iprocedure["dummy_args"] + [v for v in all_var_exprs if v not in iprocedure["dummy_args"]]
        self.global_reductions = {}
        # 
        self.kernel            = self._create_kernel_context()
    def render_cpu_routine_f03(self,launcher):
        return []
    def render_cpu_launcher_cpp(self):
        return []

class HipKernelGenerator4AcceleratorRoutine(HipKernelGenerator4CufKernel):
    def render_gpu_launcher_cpp(self,launcher):
        return []
