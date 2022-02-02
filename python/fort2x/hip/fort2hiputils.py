import addtoplevelpath
import indexer.indexerutils as indexerutils
import linemapper.linemapper as linemapper
import linemapper.linemapperutils as linemapperutils
import translator.translator as translator
import scanner.scannerutils as scannerutils
import fort2x.hip.kernelgen
import fort2x.hip.derivedtypegen
import utils.kwargs
import fort2x.hip.fort2hip

# TODO make kwargs
def create_kernel_generator_from_loop_nest(declaration_list_snippet,
                                           loop_nest_snippet,
                                           **kwargs):
    r"""Create HIP kernel generator from a declaration list snippet and
    the snippet of an directive-annotated loop nest.
    :return a code generator that can generate a HIP kernel
            and a number of different kernel launchers based on
            the original input
    :rtype: fort2x.hip.kernelgen.HipKernelGenerator4LoopNest
    :param str declaration_list_snippet: A Fortran declaration list, i.e. a number of Fortran
                                         variable and derived type declarations.
    :param str loop_nest_snippet: A Fortran loop nest annotated with directives (CUDA Fortran, OpenACC).
    :param \*\*kwargs: See below. 
    
    :Keyword Arguments:
 
    * *preproc_options* (`str`):
        C-style preprocessor options [default: '']
    * *kernel_name* (`str`): 
        A name for the kernel [default: 'mykernel']
    """
    kernel_name        = utils.kwargs.get_value("kernel_name","mykernel",**kwargs)
    kernel_hash        = utils.kwargs.get_value("kernel_hash","",**kwargs)
    preproc_options    = utils.kwargs.get_value("preproc_options",""**kwargs)
    scope              = indexerutils.create_scope_from_declaration_list(declaration_list_snippet,
                                                                         preproc_options)
    linemaps           = linemapper.read_lines(loop_nest_snippet.split("\n"),
                                               preproc_options)
    fortran_statements = linemapperutils.get_statement_bodies(linemaps)
    ttloopnest         = translator.parse_loop_kernel(fortran_statements,
                                                      scope)

    return fort2x.hip.kernelgen.HipKernelGenerator4LoopNest(kernel_name,
                                                            kernel_hash,
                                                            ttloopnest,
                                                            scope,
                                                            "\n".join(fortran_statements))

def create_interoperable_derived_type_generator(declaration_list_snippet,
                                                used_modules=[],
                                                preproc_options=""):
    """Create interoperable derived type generator from a declaration list snippet
       that describes the types.
    :return a code generator that can generate interoperable types
            from a declaration list plus routines for copying
             
    :rtype: fort2x.hip.derivedtypegen.DerivedTypeGenerator
    :param str declaration_list_snippet: A Fortran declaration list, i.e. a number of Fortran
                                         variable and derived type declarations.
    :param list used_modules: List of dicts with keys 'name' (str) and 'only' (list of str)
    :param str preproc_options: C-style preprocessor options
    """
    scope = indexerutils.create_scope_from_declaration_list(declaration_list_snippet,
                                                            preproc_options)
    return fort2x.hip.derivedtypegen.HipDerivedTypeGenerator(scope["types"],
                                                             used_modules)

def create_code_generator(**kwargs):
    r"""Create HIP Code generator from an input file and its dependencies.
    :param \*\*kw_args: See below.

    :Keyword Arguments:
        * *file_path* (``str``):
            Path to the file that should be parsed.
        * *file_content* (``str``):
            Content of the file that should be parsed.
        * *file_linemaps* (``list``):
            Linemaps of the file that should be parsed;
            see GPUFORT's linemapper component.
        * *file_is_indexed* (``bool``):
            Index already contains entries for the main file.
        * *preproc_options* (``str``): Options to pass to the C preprocessor
            of the linemapper component.
        * *other_files_paths* (``list(str)``):
            Paths to other files that contain module files that
            contain definitions required for parsing the main file.
            NOTE: It is assumed that these files have not been indexed yet.
        * *other_files_contents* (``list(str)``):
            Content that contain module files that
            contain definitions required for parsing the main file.
            NOTE: It is assumed that these files have not been indexed yet.
        * *index* (``list``):
            Index records created via GPUFORT's indexer component.
    """ 
    stree, index, linemaps = scannerutils.parse_file(**kwargs)
    print(stree)
    return fort2x.hip.fort2hip.HipCodeGenerator(stree,index,**kwargs)
