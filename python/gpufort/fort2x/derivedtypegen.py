
import copy

from . import render

from gpufort import translator

class DerivedTypeGenerator:

    def __init__(self, itypes, used_modules=[]):
        """Constructor.
        :param list itypes:       all derived type index entries for a given scope (including local derived types) 
        :param list used_modules: The used modules that should appear as use statements in the declaration list of 
                                  the copy routines.
        """
        self.itypes = copy.deepcopy(itypes)
        for itype in self.itypes:
            for ivar in itype["variables"]:
                translator.analysis.append_c_type(ivar)
        self.used_modules = used_modules
        self.synchronize_queue = "hipStreamSynchronize"
        self.error_check = "hipCheck"
        self.interop_suffix = "_interop"
        self.orig_var = "orig_type"
        self.interop_var = "interop_type"

    def render_derived_type_definitions_cpp(self):
        return [render.render_derived_types_cpp(self.itypes)]

    def render_derived_type_definitions_f03(self):
        snippets = [
            render.render_derived_types_f03(self.itypes, self.interop_suffix),
        ]
        return snippets

    def render_derived_type_routines_f03(self):
        return [
            render.render_derived_type_size_bytes_routines_f03(
                self.itypes, self.used_modules, self.interop_suffix),
            render.render_derived_type_copy_scalars_routines_f03(
                self.itypes, "in", self.used_modules, self.interop_suffix),
            render.render_derived_type_copy_scalars_routines_f03(
                self.itypes, "out", self.used_modules, self.interop_suffix),
            render.render_derived_type_copy_array_member_routines_f03(
                self.itypes, "in", "", self.used_modules,
                self.synchronize_queue, self.error_check, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_copy_array_member_routines_f03(
                self.itypes, "out", "", self.used_modules,
                self.synchronize_queue, self.error_check, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_init_array_member_routines_f03(
                self.itypes, "", self.used_modules, self.error_check,
                self.interop_suffix, self.orig_var, self.interop_var),
            render.render_derived_type_destroy_array_member_routines_f03(
                self.itypes, "", self.used_modules, self.synchronize_queue,
                self.error_check, self.interop_suffix, self.orig_var,
                self.interop_var),
            render.render_derived_type_copy_array_member_routines_f03(
                self.itypes, "in", "_async", self.used_modules,
                self.error_check, self.synchronize_queue, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_copy_array_member_routines_f03(
                self.itypes, "out", "_async", self.used_modules,
                self.synchronize_queue, self.error_check, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_init_array_member_routines_f03(
                self.itypes, "_async", self.used_modules, self.error_check,
                self.interop_suffix, self.orig_var, self.interop_var),
            render.render_derived_type_destroy_array_member_routines_f03(
                self.itypes, "_async", self.used_modules,
                self.synchronize_queue, self.error_check, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_destroy_routines_f03(
                self.itypes, "", self.used_modules, self.interop_suffix,
                self.orig_var, self.interop_var),
            render.render_derived_type_destroy_routines_f03(
                self.itypes, "_async", self.used_modules, self.interop_suffix,
                self.orig_var, self.interop_var),
        ]
