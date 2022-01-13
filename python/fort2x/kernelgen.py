import addtoplevelpath
import indexer.scoper as scoper
import utils.logging 

LOG_PREFIX = "fort2x.hip.kernelgen.KernelGeneratorBase"

class KernelGeneratorBase:
    @staticmethod
    def strip_member_access(var_exprs):
        """Strip off member access parts from derived type member access expressions,
           e.g. 'a%b%c' becomes 'a'.
        """
        result = []
        return [var_expr.split("%",maxsplit=1)[0] for var_expr in var_exprs]
    @staticmethod
    def lookup_index_vars(var_exprs,
                          consumed_var_exprs=[]):
        """Search scope for index vars and remove corresponding 
           var expression from all_vars2 list."""
        ivars              = []
        for var_expr in var_exprs:
            ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
            ivars.append(ivar)
            consumed_var_exprs.append(var_expr)
        return ivars
    @staticmethod
    def lookup_index_entries_for_vars_in_kernel_body(scope,
                                                     all_vars,
                                                     reductions,
                                                     shared_vars,
                                                     local_vars,
                                                     error_handling=None):
        """Lookup index variables
        :param list all_vars: List of all variable expressions (var)
        :param list reductions: List of tuples pairing a reduction operation with the associated
                                 variable expressions
        :param list shared:     List of variable expressions that are shared by the workitems/threads in a workgroup/threadblock
        :param list local_vars: List of variable expressions that can be mapped to local variables per workitem/thread
        :param str error_handling: Emit error messages if set to 'strict'; else emit warning
        :note: Emits errors (or warning) if a variable in another list is not present in all_vars
        :note: Emits errors (or warning) if a reduction variable is part of a struct.
        :return: A tuple containing (in this order): global variables, reduced global variables, shared variables, local variables
                 as list of index entries. The reduced global variables have an extra field 'op' that
                 contains the reduction operation.
        """
        # TODO parse bounds and right-hand side expressions here too
        utils.logging.log_enter_function(LOG_PREFIX,"lookup_index_entries_for_variables")
    
        consumed  = []
        ilocal_vars  = KernelGeneratorBase.lookup_index_vars(KernelGeneratorBase.strip_member_access(local_vars),
                                                             consumed)
        ishared_vars = KernelGeneratorBase.lookup_index_vars(KernelGeneratorBase.strip_member_access(shared_vars),
                                                             consumed)
        all_vars2 = [v for v in KernelGeneratorBase.strip_member_access(all_vars) if not v in consumed]
     
        rglobal_reduced_vars = []
        iglobal_vars         = []
    
        for reduction_op,var_exprs in reductions.items():
            for var_expr in var_exprs:
                if "%" in var_expr:
                   if error_handling=="strict":
                       pass # TODO error
                   else:
                       pass # TOOD warn
                else:
                   ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
                   rvar    = copy.deepcopy(ivar)
                   rvar["op"] = reduction_op
                   rglobal_reduced_vars.append(rvar)
                try:
                    all_vars2.remove(var_expr)
                except:
                    pass # TODO error
        for var_expr in all_vars2:
            ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
            iglobal_vars.append(ivar)
    
        utils.logging.log_leave_function(LOG_PREFIX,"lookup_index_entries_for_variables")
        return iglobal_vars, rglobal_reduced_vars, ishared_vars, ilocal_vars
    
    def _create_kernel_base_context(self,
                                    kernel_name,
                                    c_body,
                                    launch_bounds=None):
        #global GET_LAUNCH_BOUNDS
        #launch_bounds = GET_LAUNCH_BOUNDS(kernel_name)
        return {
               "name"                : kernel_name,
               "launch_bounds"       : "" if launch_bounds == None else launch_bounds,
               "c_body"              : c_body,
               "global_vars"         : [],
               "global_reduced_vars" : [],
               "shared_vars"         : [],
               "local_vars"          : [],
               }
    def _create_launcher_base_context(self,
                                      kernel_name,
                                      kind,
                                      problem_size,
                                      debug_output):
        return {
               "kind"         : kind,
               "name"         : "_".join(["launch",kernel_name,kind]),
               "used_modules" : [],
               "problem_size" : problem_size,
               "debug_output" : debug_output,
               }
    def render_gpu_kernel_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_begin_kernel_comment_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_end_kernel_comment_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_begin_kernel_comment_f03(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_end_kernel_comment_f03(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_cpu_launcher_cpp(self,launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_launcher_interface_f03(self,launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_cpu_routine_f03(self,launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
    def render_gpu_kernel_cpp(self):
        """:return: Snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []
