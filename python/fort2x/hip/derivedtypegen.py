import addtoplevelpath

import fort2x.hip.render
import indexer.scoper as scoper

class DerivedTypeGenerator:
    def __init__(itypes,
                 used_modules=[]):
        """Constructor.
        :param list itypes:       all derived type index entries for a given scope (including local derived types) 
        :param list itypes_local: all derived types in the local procedure or program scope. They must
                                  be recreated in the declaration list of the copy functions.
        :param list used_modules: The used modules that should appear as use statements in the declaration list of 
                                  the copy routines.
        """
        self.itypes         = itypes
        self.interop_suffix = "_interop"
        self.orig_var       = "orig_type",
        self.interop_var    = "interop_type",
    def render_derived_type_definitions_cpp(self):
        return [fort2x.hip.render.render_derived_types_cpp(types)]
    def render_derived_type_definitions_f03(self):
        snippets = [
                   fort2x.hip.render.render_derived_types_f03(self.itypes,
                                                           self.interop_suffix),
                   ]
        return snippets 
    def render_derived_type_routines_f03(self):
        return [
               fort2x.hip.render.render_derived_type_size_bytes_routines_f03(self.itypes,
                                                                          self.interop_suffix,
                                                                          self.used_modules),
               fort2x.hip.render.render_derived_type_copy_scalars_routines_f03(self.itypes,
                                                                            self.interop_suffix,
                                                                            self.used_modules),
               fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self.itypes,
                                                                                 self.interop_suffix,
                                                                                 self.orig_var,
                                                                                 self.interop_var,
                                                                                 self.used_modules),
               ]
