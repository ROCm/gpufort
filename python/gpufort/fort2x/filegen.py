# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from . import render


class FileGenerator():

    def generate_code(self):
        assert False, "Not implemented!"
        return ""

    def generate_file(self, file_path):
        with open(file_path, "w") as outfile:
            outfile.write(self.generate_code())


class CppFileGenerator(FileGenerator):
    PROLOG = ""

    # TODO do not just write but read and replace certain
    # code sections; make aware of kernel names and their hash value
    def __init__(self, **kwargs):
        r"""Constructor.
       :param \*\*kwargs: See below.

       :Keyword Arguments:

       * *guard* (`str`):
           Include guard symbol.
       * *prolog* (`str`):
           A prolog to put at the top of the file.
       * *includes_prolog* (`str`):
           A prolog to put before the includes.
       * *includes_epilog* (`str`):
           An epilog to put below the includes.
       * *default_includes* (`list`):
           The default includes.
       * *emit_only_types* (`str`):
           Only types and no kernels and no launchers
           shall be written into the generated file.
       * *emit_only_kernels* (`str`):
           Only kernels and types, as they might be required by the kernels,
           shall be written into the generated file.
       """
        util.kwargs.set_from_kwargs(self, "guard", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "prolog", "", **kwargs)
        self.prolog = CppFileGenerator.PROLOG + self.prolog
        util.kwargs.set_from_kwargs(self, "includes_prolog", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "includes_epilog", "", **kwargs)
        util.kwargs.set_from_kwargs(self, "default_includes", [], **kwargs)
        util.kwargs.set_from_kwargs(self, "emit_only_types", False, **kwargs)
        util.kwargs.set_from_kwargs(self, "emit_only_kernels", False, **kwargs)
        #
        self.rendered_types = []
        self.rendered_kernels = []
        self.rendered_launchers = []
        self.includes = []
        #
        pass

    def stores_any_code_or_includes(self):
        return len(self.snippets)\
               or len(self.includes)

    def merge(self, other):
        """Merge two file generator instances.
        Only adds (non-default) includes of other file generator if they are not present yet."""
        self.rendered_launchers += other.rendered_launchers
        self.rendered_types += other.rendered_types
        self.rendered_kernels += other.rendered_kernels
        for include in other.includes:
            if include not in self.includes:
                self.includes.append(include)

    def generate_code(self):
        """
        :param bool only_types: Only write rendered derived types (=structs) into C++ file.
        :param bool only_kernels: Only write rendered kernels (and derived types) into C++ file.
        """
        snippets = self.rendered_types
        if not self.emit_only_types:
            snippets += self.rendered_kernels
            if not self.emit_only_kernels:
                snippets += self.rendered_launchers
        return render.render_c_file_cpp(self.guard, snippets,
                                        self.default_includes + self.includes,
                                        self.prolog, self.includes_prolog,
                                        self.includes_epilog)


class FortranModuleGenerator(FileGenerator):
    PROLOG = ""

    def __init__(self, **kwargs):
        r"""Constructor.
       :param \*\*kwargs: See below.

       :Keyword Arguments:

       * *prolog* (`str`):
           Prolog to put at the top of the file.
       * *name* (`str`):
           Name to give the module [default: 'mymodule'].
       * *default_used_modules* (`list`):
           The default modules that the generated interfaces
           and routines shall use [default: []].
       """
        util.kwargs.set_from_kwargs(self, "prolog", "", **kwargs)
        self.prolog = FortranModuleGenerator.PROLOG + self.prolog
        util.kwargs.set_from_kwargs(self, "name", "mymodule", **kwargs)
        util.kwargs.set_from_kwargs(self, "default_used_modules", [], **kwargs)
        #
        self.used_modules = []
        self.rendered_types = []
        self.rendered_interfaces = []
        self.rendered_routines = []
        pass

    def stores_any_code(self):
        return len(self.rendered_types)\
               or len(self.rendered_interfaces)\
               or len(self.rendered_routines)

    def merge(self, other):
        """Merge two file generator instances."""
        self.used_modules += other.used_modules
        self.rendered_types += other.rendered_types
        self.rendered_interfaces += other.rendered_interfaces
        self.rendered_routines += other.rendered_routines

    def generate_code(self):
        return render.render_interface_module_f03(self.name, self.used_modules,
                                                  self.rendered_types,
                                                  self.rendered_interfaces,
                                                  self.rendered_routines,
                                                  self.prolog)
