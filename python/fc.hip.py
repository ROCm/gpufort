#!/usr/bin/env python3

"""
Wrapper compiler based on GPUFORT's
HIP C++ code generation capabilities.

:Long-term ToDos:

* Capture error output when compiling modified file
  and C++ file and relate to original input file.
"""
import os
import sys
import textwrap
import re
import subprocess
import argparse

import converter
import opts

from gpufort import util
from gpufort import scanner
from gpufort import linemapper

def populate_cl_arg_parser(parser):
    # General options
    parser.add_argument("inputs",
                        help="The input files",
                        type=str,
                        nargs="+",
                        default=[])
    parser.add_argument("-c",
                        dest="only_compile",
                        action="store_true",
                        help="Only compile; do not link.")
    parser.add_argument("-o",
                        dest="output",
                        default=None,
                        type=str,
                        help="Path/name for the result.")
    parser.add_argument(
        "--gpufort-fc",
        dest="fortran_compiler",
        default="hipfc",
        type=str,
        help=
        "Compiler for Fortran code, defaults to HIPFORT's gfortran-based 'hipfc'. Use '--gpufort-fcflags <flags>' to set compilation flags and '--gpufort-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and HIP C++ compiler and their respective options must be specified after all other GPUFORT options.",
    )
    parser.add_argument(
        "--gpufort-cc",
        dest="cpp_compiler",
        default="hipcc",
        type=str,
        help=
        "Compiler for HIP C++ code, defaults to 'hipcc'. Use '--gpufort-cflags <flags>' to set compilation flags and '--gpufort-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and HIP C++ compiler and their respective options must be specified after all other GPUFORT options.",
    )
    parser.add_argument(
        "--gpufort-ldflags",
        dest="gpufort_ldflags",
        default="hipcc",
        type=str,
        help=
        "Compiler for HIP C++ code, defaults to 'hipcc'. Use '--gpufort-cflags <flags>' to set compilation flags and '--gpufort-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and HIP C++ compiler and their respective options must be specified after all other GPUFORT options.",
    )
    parser.add_argument(
        "-save-temps",
        dest="save_temps",
        action="store_true",
        help=
        "Save temporary output files, i.e. intermediate results produced by the translation of the original code to standard Fortran plus HIP C++.",
    )
    parser.add_argument(
        "-only-codegen",
        dest="only_codegen",
        action="store_true",
        help=
        "Only generate code. Do not compile.",
    )
    parser.add_argument(
        "-no-codegen",
        dest="no_codegen",
        action="store_true",
        help=
        "Do not generate code. This assumes that this has been done before.",
    )
    parser.set_defaults(only_compile=False,
                        only_codegen=False,
                        no_codegen=False)

def check_inputs(paths):
    """:return: Typical object file name/path that compilers derive from a Fortran/C++ source file.
    :raise util.error.ValueError: If no single file with Fortran file extension could be identified
                       and the other inputs do not have an '.so', '.a', or '.o' ending.
    """
    fortran_files = []
    objects = []
    others = []
    for path in paths:
        abspath = os.path.abspath(path)
        if re.match(r".+\.(f[0-9]*)$", path, re.IGNORECASE):
            fortran_files.append(abspath)
        elif re.match(r".+\.(so|a|o)$", path, re.IGNORECASE):
            objects.append(abspath)
        else:
            others.append(path)
    if len(fortran_files) > 1:
        raise ValueError(
            "specify only a single Fortran file as input; have: "
            + ",".join(fortran_files))
    if len(others):
        raise ValueError(
            "The following files are neither Fortran nor object files: "
            + ",".join(others))
    return fortran_files[0], objects


def get_object_path(source_path):
    """:return: Typical object file name/path that compilers derive from a Fortran/C++ source file.
    """
    result = re.sub(r"\.f[0-9]*$", ".o", source_path, re.IGNORECASE)
    result = re.sub(r"\.(c|cc|cxx|cpp)$", ".cpp.o", result, re.IGNORECASE)
    return result


def handle_subprocess_output(triple):
    status, output, errors = triple
    sys.stdout.write(output)
    sys.stderr.write(errors)
    # TODO detect errors and warnings line numbers
    # TODO use single-line-block linemaps to trace back source of errors
    # in original file

def hardcode_pure_converter_opts():
    """Fixes certain opts, makes them unaffected by config.
    """
    # always overwrite these options
    scanner.opts.destination_dialect = "hipgpufort"
    linemapper.opts.line_grouping_ifdef_macro = None
    #opts.only_create_gpufort_module_files = False
    #opts.skip_create_gpufort_module_files = False
    opts.only_modify_translation_source = False
    opts.only_emit_kernels_and_launchers = False
    opts.only_emit_kernels = False

def remove_file(path):
    """Tries to remove file. Throws no error if it doesn't exists."""
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    config_file_path, include_dirs, defines, fortran_and_cpp_compiler_options = converter.parse_raw_cl_args()
    if config_file_path != None:
        parse_config(config_file_path)
    # parse command line arguments
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
(EXPERIMENTAL) Wrapper compiler for CUDA and OpenACC Fortran programs based on GPUFORT's HIP C++ code generator.

NOTE: This tool is based on the GPUFORT converter tool. Options inherited from the latter
have a '--' prefix while this tool's options have a '-' prefix."""))
    populate_cl_arg_parser(parser)
    converter.populate_cl_arg_parser(parser, False)
    args, unknown_args = converter.parse_cl_args(parser, False)
    
    if args.only_codegen and args.no_codegen:
        util.logging.log_error(opts.log_prefix,"__main__","'-only-codegen' and '-no-codegen' are mutually exclusive")
        sys.exit(2)

    converter.set_include_dirs(args.working_dir,
                               args.search_dirs + include_dirs)
    converter.map_args_to_opts(args,opts.include_dirs,defines,fortran_and_cpp_compiler_options,False)
    hardcode_pure_converter_opts()
    if len(opts.post_cli_actions):
        msg = "run registered actions"
        util.logging.log_info(msg, verbose=False)
        for action in opts.post_cli_actions:
            if callable(action):
                action(args, unknown_args)

    try:
        infile_path, object_files = check_inputs(args.inputs)
    except ValueError as uve:
        util.logging.log_error(opts.log_prefix, "__main__", str(uve))
        sys.exit(2)
    log_file_path = converter.init_logging(infile_path)

    modified_fortran_file_path = "".join([infile_path,opts.fortran_file_ext])
    cpp_file_path = "".join([infile_path,opts.cpp_file_ext])
    if args.only_codegen or not args.no_codegen:
        # configure scanner source dialects based on -openacc flag
        if not args.openacc and "acc" in scanner.opts.source_dialects:
            scanner.opts.source_dialects.remove("acc")
        elif args.openacc:
            scanner.opts.source_dialects.add("acc")
        converter.run_checked(infile_path,modified_fortran_file_path,cpp_file_path,defines)

    # output file name
    if not args.only_codegen:
        outfile_path = args.output
        if outfile_path == None:
            if args.only_compile:
                outfile_path = get_object_path(
                    infile_path) # ! object file should be named after object file
            else:
                outfile_path = os.path.join(os.path.dirname(infile_path), "a.out")

        # flags
        fortran_cflags = []
        cpp_cflags = []
        ldflags = []
        if args.use_default_flags:
            cpp_cflags += converter.get_basic_cflags()
            fortran_cflags += list(cpp_cflags)
            ldflags += converter.get_basic_ldflags(args.openacc)
            if os.path.basename(args.fortran_compiler) == "gfortran":
                fortran_cflags = converter.get_gfortran_cflags()
            if os.path.basename(args.cpp_compiler) in ["hipcc", "hipcc.bin"]:
                cpp_cflags += converter.get_hipcc_cflags()

        # cpp compilation
        cpp_object_path = get_object_path(cpp_file_path)
        cpp_cmd = ([args.cpp_compiler] 
                   + cpp_cflags + fortran_and_cpp_compiler_options["-cflags"]
                   + ["-c", cpp_file_path]
                   + ["-o", cpp_object_path])
        handle_subprocess_output(util.subprocess.run_subprocess(cpp_cmd,args.verbose))
        # fortran compilation
        modified_object_path = get_object_path(modified_fortran_file_path)
        # emit a single object combining the Fortran and C++ objects
        if args.only_compile:
            fortran_cmd = ([args.fortran_compiler] + fortran_cflags
                           + fortran_and_cpp_compiler_options["-fcflags"]
                           + ["-c", modified_fortran_file_path]
                           + ["-o", modified_object_path])
            handle_subprocess_output(util.subprocess.run_subprocess(fortran_cmd,args.verbose))
            # merge object files; produces: outfile_path
            handle_subprocess_output(
                util.subprocess.run_subprocess([
                    "ld", "-r", "-o", outfile_path, modified_object_path,
                    cpp_object_path
                ],args.verbose))
            os.remove(modified_object_path)
        # emit an executable
        else:
            fortran_cmd = ([args.fortran_compiler] + fortran_cflags
                           + fortran_and_cpp_compiler_options["-fcflags"]
                           + [modified_fortran_file_path] 
                           + [cpp_object_path] 
                           + ldflags
                           + fortran_and_cpp_compiler_options["-ldflags"]
                           + ["-o", outfile_path])
            handle_subprocess_output(util.subprocess.run_subprocess(fortran_cmd,args.verbose))
            # merge object files; produces: outfile_path
            remove_file(cpp_object_path)
        if not args.save_temps:
            remove_file(cpp_file_path)
            remove_file(modified_fortran_file_path)

    converter.shutdown_logging(log_file_path)
else:
    assert False, "This module is supposed to be used as standalone application!"
