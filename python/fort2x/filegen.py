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
    def __init__(self
                 guard           
                 prolog          = ""
                 includes_prolog = ""
                 includes_epilog = ""):
       self.guard           = guard
       self.prolog          = PROLOG + prolog
       self.includes_prolog = includes_prolog
       self.includes_epilog = includes_epilog
       self.snippets        = []
       self.includes        = []
       pass
    def merge(self,
              other):
        self.snippets        += other.snippets
        self.includes        += other.includes 
    def generate_code(self):
        return fort2x.render.render_c_file_cpp(self.guard,
                                               self.snippets,
                                               self.includes,
                                               self.prolog,
                                               self.includes_prolog,
                                               self.includes_epilog)

class FortranModuleGenerator(FileGenerator):
    PROLOG = ""

    def __init__(self,
                 name
                 prolog = ""):
       self.name                = name
       self.prolog              = PROLOG + prolog
       self.used_modules        = []
       self.rendered_types      = []
       self.rendered_interfaces = []
       self.rendered_routines   = [] 
       pass 
    def merge(self,
              other):
        self.used_modules        += other.used_modules
        self.rendered_types      += other.rendered_types
        self.rendered_interfaces += other.rendered_interfaces
        self.rendered_routines   += other.rendered_routines
    def generate_code(self):
        return fort2x.render.render_interface_module_f03(self.name,
                                                         self.used_modules,
                                                         self.rendered_types,
                                                         self.rendered_interfaces,
                                                         self.rendered_routines,
                                                         self.prolog)
