import addtoplevelpath

import fort2x.hip.render
import indexer.scoper as scoper

class DerivedTypeGenerator:
    def __init__(self,
                 itypes,
                 used_modules=[{"name" : mod,"only" : []} for mod in ["hipfort","hipfort_check","gpufort_array"]]):
        """Constructor.
        :param list itypes:       all derived type index entries for a given scope (including local derived types) 
        :param list used_modules: The used modules that should appear as use statements in the declaration list of 
                                  the copy routines.
        """
        self.itypes            = itypes
        self.used_modules      = used_modules
        self.synchronize_queue = "hipStreamSynchronize"
        self.error_check       = "hipCheck"
        self.interop_suffix    = "_interop"
        self.orig_var          = "orig_type"
        self.interop_var       = "interop_type"
    def render_derived_type_definitions_cpp(self):
        return [fort2x.hip.render.render_derived_types_cpp(self.itypes)]
    def render_derived_type_definitions_f03(self):
        snippets = [
                   fort2x.hip.render.render_derived_types_f03(self.itypes,
                                                           self.interop_suffix),
                   ]
        return snippets 
    def render_derived_type_routines_f03(self):
        return [
               fort2x.hip.render.render_derived_type_size_bytes_routines_f03(self.itypes,
                                                                             self.used_modules,
                                                                             self.interop_suffix),
               fort2x.hip.render.render_derived_type_copy_scalars_routines_f03(self.itypes,
                                                                               "in",
                                                                               self.used_modules,
                                                                               self.interop_suffix),
               fort2x.hip.render.render_derived_type_copy_scalars_routines_f03(self.itypes,
                                                                               "out",
                                                                               self.used_modules,
                                                                               self.interop_suffix),
               fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self.itypes,
                                                                                    "in",
                                                                                    "",
                                                                                    self.used_modules,
                                                                                    self.synchronize_queue,
                                                                                    self.error_check,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self.itypes,
                                                                                    "out",
                                                                                    "",
                                                                                    self.used_modules,
                                                                                    self.synchronize_queue,
                                                                                    self.error_check,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_init_array_member_routines_f03(self.itypes,
                                                                                    "",
                                                                                    self.used_modules,
                                                                                    self.error_check,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_destroy_array_member_routines_f03(self.itypes,
                                                                                       "",
                                                                                       self.used_modules,
                                                                                       self.synchronize_queue,
                                                                                       self.error_check,
                                                                                       self.interop_suffix,
                                                                                       self.orig_var,
                                                                                       self.interop_var),
               fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self.itypes,
                                                                                    "in",
                                                                                    "_async",
                                                                                    self.used_modules,
                                                                                    self.error_check,
                                                                                    self.synchronize_queue,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self.itypes,
                                                                                    "out",
                                                                                    "_async",
                                                                                    self.used_modules,
                                                                                    self.synchronize_queue,
                                                                                    self.error_check,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_init_array_member_routines_f03(self.itypes,
                                                                                    "_async",
                                                                                    self.used_modules,
                                                                                    self.error_check,
                                                                                    self.interop_suffix,
                                                                                    self.orig_var,
                                                                                    self.interop_var),
               fort2x.hip.render.render_derived_type_destroy_array_member_routines_f03(self.itypes,
                                                                                       "_async",
                                                                                       self.used_modules,
                                                                                       self.synchronize_queue,
                                                                                       self.error_check,
                                                                                       self.interop_suffix,
                                                                                       self.orig_var,
                                                                                       self.interop_var),
               fort2x.hip.render.render_derived_type_destroy_routines_f03(self.itypes,
                                                                          "",
                                                                          self.used_modules,
                                                                          self.interop_suffix,
                                                                          self.orig_var,
                                                                          self.interop_var),
               fort2x.hip.render.render_derived_type_destroy_routines_f03(self.itypes,
                                                                          "_async",
                                                                          self.used_modules,
                                                                          self.interop_suffix,
                                                                          self.orig_var,
                                                                          self.interop_var),
               ]
