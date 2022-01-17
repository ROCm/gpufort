import addtoplevelpath
import indexer.indexerutils as indexerutils
import linemapper.linemapper as linemapper
import linemapper.linemapperutils as linemapperutils
import translator.translator as translator
import fort2x.hip.kernelgen

def create_kernel_generator_from_loop_nest(declaration_list_snippet,
                                           loop_nest_snippet,
                                           kernel_name="mykernel",
                                           kernel_hash=""):
    """
    Create HIP kernel generator from a declaration list snippet and
    the snippet of an directive-annotated loop nest.
    :return a code generator that can generate a HIP kernel
            and a number of different kernel launchers based on
            the original input
    :rtype: fort2x.hip.kernelgen.HipKernelGenerator4LoopNest
    :param str declaration_list_snippet: A Fortran declaration list, i.e. a number of Fortran
                                         variable and derived type declarations.
    :param str loop_nest_snippet: A Fortran loop nest annotated with directives (CUDA Fortran, OpenACC).
    :param str kernel_name: The name you want to give the kernel
    """
    scope              = indexerutils.create_scope_from_declaration_list(declaration_list_snippet)
    linemaps           = linemapper.read_lines(loop_nest_snippet.split("\n"))
    fortran_statements = linemapperutils.get_statement_bodies(linemaps)
    ttloopnest         = translator.parse_loop_kernel(fortran_statements,
                                                      scope)

    return fort2x.hip.kernelgen.HipKernelGenerator4LoopNest(kernel_name,
                                                            kernel_hash,
                                                            ttloopnest,
                                                            scope,
                                                            "\n".join(fortran_statements))
