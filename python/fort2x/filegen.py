import fort2x.render

class FileGenerator():
    def generate_code(self):
        assert False, "Not implemented!"
        return ""
    def generate_file(self,filepath):
        with open(filepath,"w") as outfile:
            outfile.write(self.generate_code())

class CPPFileGenerator(FileGenerator):
    PROLOG = ""

    # TODO do not just write but read and replace certain
    # code sections; make aware of kernel names and their hash value
    def __init__(self,
                 guard,
                 prolog          = "",
                 includes_prolog = "",
                 includes_epilog = ""):
       self.guard           = guard
       self.prolog          = CPPFileGenerator.PROLOG + prolog
       self.includes_prolog = includes_prolog
       self.includes_epilog = includes_epilog
       # 
       self.rendered_types     = []
       self.rendered_kernels   = []
       self.rendered_launchers = []
       self.default_includes   = []
       self.includes           = []
       #
       self.emit_only_types   = False 
       self.emit_only_kernels = False 
       pass
    def stores_any_code_or_includes(self):
        return len(self.snippets)\
               or len(self.includes)
    def merge(self,
              other):
        """Merge two file generator instances.
        Only adds (non-default) includes of other file generator if they are not present yet."""
        self.rendered_launchers += other.rendered_launchers
        self.rendered_types     += other.rendered_types    
        self.rendered_kernels   += other.rendered_kernels  
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
        return fort2x.render.render_c_file_cpp(self.guard,
                                               snippets,
                                               self.default_includes+self.includes,
                                               self.prolog,
                                               self.includes_prolog,
                                               self.includes_epilog)

class FortranModuleGenerator(FileGenerator):
    PROLOG = ""

    def __init__(self,
                 name,
                 prolog = ""):
       self.name                 = name
       self.prolog               = FortranModuleGenerator.PROLOG + prolog
       self.default_used_modules = []
       self.used_modules         = []
       self.rendered_types       = []
       self.rendered_interfaces  = []
       self.rendered_routines    = [] 
       pass 
    def stores_any_code(self):
        return len(self.rendered_types)\
               or len(self.rendered_interfaces)\
               or len(self.rendered_routines)
    def merge(self,
              other):
        """Merge two file generator instances."""
        self.used_modules        += other.used_modules
        self.rendered_types      += other.rendered_types
        self.rendered_interfaces += other.rendered_interfaces
        self.rendered_routines   += other.rendered_routines
    def generate_code(self):
        return fort2x.render.render_interface_module_f03(self.name,
                                                         self.default_used_modules+self.used_modules,
                                                         self.rendered_types,
                                                         self.rendered_interfaces,
                                                         self.rendered_routines,
                                                         self.prolog)
