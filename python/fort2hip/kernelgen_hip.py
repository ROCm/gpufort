import addtoplevelpath

import fort2hip.kernelgen
import fort2hip.render

class HipKernelGeneratorBase(fort2hip.kernelgen.KernelGeneratorBase):
    """
    """
    def __init__(self,
                 kernel_name,
                 kernel_hash,
                 scope,
                 fortran_snippet="",
                 error_handling=None):
        self.kernel_name     = kernel_name
        self.kernel_hash     = kernel_hash
        self.scope           = scope
        self.fortran_snippet = fortran_snippet
        self.error_handling  = error_handling 
        # to be set by subclass:
        self.c_body                = ""
        self.all_vars              = []
        self.global_reductions     = {}
        self.shared_vars           = []
        self.local_vars            = []
    def _create_kernel_context(self):
        kernel = self._create_kernel_base_context(self.kernel_name,
                                                  self.c_body)
        kernel["global_vars"],
        kernel["global_reduced_vars"],
        kernel["shared_vars"],
        kernel["local_vars"] = fort2hip.kernelgen.KernelGeneratorBase.lookup_index_entries_for_vars_in_kernel_body(self.scope,
                                                                                                                   self.all_vars,
                                                                                                                   self.global_reductions,
                                                                                                                   self.shared_vars,
                                                                                                                   self.local_vars,
                                                                                                                   self.error_handling)
        return kernel
    def __create_launcher_base_context(self,
                                              kind,
                                              debug_output,
                                              problem_size):
        """Create base context for kernel launcher code generation."""
        return {
               "kind"         : kind,
               "name"         : "_".join(["launch",self.kernel_name,kind]),
               "used_modules" : [],
               "problem_size" : problem_size,
               "debug_output" : debug_output,
               }
    def create_launcher_context(self,
                                       kind,
                                       debug_output,
                                       problem_size,
                                       grid  = [],
                                       block = []):
        """Create context for HIP kernel launcher code generation.
        :param str kind:   
        :param list grid:  d-dimensional list of string expressions for the grid dimensions
        :param list block: d-dimensional list of string expressions for the block dimensions
        :note: Parameters grid & block are only required if the kind is set to "hip".
        """
        kernel_launcher = self.__create_launcher_base_context(kind,
                                                              debug_output,
                                                              problem_size)
        if kind == "hip":
            kernel_launcher["grid"]  = grid
            kernel_launcher["block"] = block
        return kernel_launcher
    def render_gpu_kernel_cpp(self):
        return [
               fort2hip.render.render_hip_kernel_comment_cpp(self.fortran_snippet),
               fort2hip.render.render_hip_kernel_cpp(self.kernel),
               ]
    def render_begin_kernel_comment_cpp(self):
        return [fort2hip.render.render_begin_kernel_cpp(self.kernel)]
    def render_end_kernel_comment_cpp(self):
        return [fort2hip.render.render_end_kernel_cpp(self.kernel)]
    def render_gpu_launcher_cpp(kernel_launcher):
        return [fort2hip.render.render_hip_kernel_launcher_cpp(self.kernel,
                                                               kernel_launcher)]
    def render_cpu_launcher_cpp(self):
        return [
               fort2hip.render.render_cpu_routine_cpp(self.kernel),
               fort2hip.render.render_cpu_launcher_cpp(self.kernel,kernel_launcher),
               ]
    def render_launcher_interface_f03(kernel_launcher):
        return [fort2hip.render.render_launcher_interface_f03(self.kernel,kernel_launcher)]
    def render_cpu_routine_f03(kernel_launcher):
        return [fort2hip.render.render_cpu_routine_f03(self.kernel,kernel_launcher)]

# derived classes
class HipKernelGenerator4LoopNest(HipKernelGeneratorBase):
    def __init__(self,
                 kernel_name,
                 kernel_hash,
                 ttloopnest,
                 scope,
                 fortran_snippet="",
                 error_handling=None):
        HipKernelGeneratorBase.__init__(self,
                                        kernel_name,
                                        kernel_hash,
                                        scope,
                                        fortran_snippet,
                                        error_handling)
        self.all_vars          = ttloopnest.vars_in_body()
        self.global_reductions = ttloopnest.gang_team_reductions()
        self.shared_vars       = ttloopnest.gang_team_shared_vars()
        self.local_vars        = ttloopnest.local_scalars()
        self.c_body            = ttloopnest.c_str()
        # 
        self.kernel            = self._create_kernel_context()

class HipKernelGenerator4CufKernel(HipKernelGeneratorBase):
    def __init__(self,
                 procedure_name,
                 procedure_hash,
                 ttprocedure,
                 iprocedure,
                 scope,
                 fortran_snippet=None,
                 error_handling=None):
        HipKernelGeneratorBase.__init__(self,
                                        procedure_name,
                                        
                                        scope,
                                        fortran_snippet,
                                        error_handling)
        self.shared_vars       = [ivar["name"] for ivar in iprocedure["variables"] if "shared" in ivar["qualifiers"]]
        self.local_vars        = [ivar["name"] for ivar in iprocedure["variables"] if ivar["name"] not in iprocedure["dummy_args"]]
        all_var_exprs          = self._strip_member_access(ttprocedure.vars_in_body()) # in the body, there might be variables present from used modules
        self.all_vars          = iprocedure["dummy_args"] + [v for v in all_var_exprs if v not in iprocedure["dummy_args"]]
        self.global_reductions = {}
        # 
        self.kernel            = self._create_kernel_context()
    def render_cpu_routine_f03(kernel_launcher):
        return []
    def render_cpu_launcher_cpp(self):
        return []

class HipKernelGenerator4AcceleratorRoutine(HipKernelGenerator4CufKernel):
    def render_gpu_launcher_cpp(kernel_launcher):
        return []
