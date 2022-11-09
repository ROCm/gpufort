#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import io
import pathlib
import argparse
import hashlib
import cProfile
import pstats
import textwrap
import shlex

# local imports
from gpufort import util
from gpufort import scanner
from gpufort import indexer
from gpufort import linemapper
from gpufort import translator
from gpufort import fort2x

import opts

__GPUFORT_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
__GPUFORT_ROOT_DIR = os.path.abspath(os.path.join(__GPUFORT_PYTHON_DIR, ".."))

def parse_raw_cl_compiler_args():
    """Collect fortran and HIP C++ compiler options.
    :return: list of fortran and HIP C++ compiler options
             plus the index where one was found the first time. 
    """
    options = sys.argv[1:]

    prefix = "--gpufort"
    opts   = ["fc", "fcflags","cc","cflags"]
    compiler_options = ["-".join([prefix, x]) for x in opts]
    compiler_options.append("-".join([prefix,"ldflags"]))
    #
    result = {}
    for copt in compiler_options:
        result[copt.replace(prefix,"")] = []
    copt = None
    first_arg_index = len(options) + 1
    for i, opt in enumerate(list(options)):
        if opt in compiler_options:
            copt = opt
            first_arg_index = min(i+1,first_arg_index)
        else:
            if copt != None:
                result[copt.replace(prefix,"")].append(opt)
    sys.argv = sys.argv[:first_arg_index]
    return result

def parse_raw_cl_args():
    """Parse command line arguments before using argparse.
    Allows the loaded config file to modify
    the argparse switches and help information.
    Further transform some arguments.
    """
    config_file_path = None
    working_dir_path = os.getcwd()
    include_dirs = []
    defines = []
    
    compiler_args = parse_raw_cl_compiler_args()
    options = sys.argv[1:]
    for i, opt in enumerate(list(options)):
        if opt == "--working-dir":
            if i + 1 < len(options):
                working_dir_path = options[i + 1]
        elif opt == "--config-file":
            if i + 1 < len(options):
                config_file_path = options[i + 1]
        elif opt.startswith("-I"):
            include_dirs.append(opt[2:])
            sys.argv.remove(opt)
        elif opt.startswith("-D"):
            defines.append(opt)
            sys.argv.remove(opt)
    if not os.path.exists(working_dir_path):
        msg = "working directory '{}' cannot be found".format(working_dir_path)
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    if config_file_path != None:
        if config_file_path[0] != "/":
            config_file_path = working_dir_path + "/" + config_file_path
        if not os.path.exists(config_file_path):
            msg = "config file '{}' cannot be found".format(config_file_path)
            print("ERROR: " + msg, file=sys.stderr)
            sys.exit(2)
    return config_file_path, include_dirs, defines, compiler_args

def parse_config(config_file_path):
    """Load user-supplied config file."""
    gpufort_python_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        if os.path.isabs(config_file_path.strip()):
            CONFIG_FILE = config_file_path
        else:
            CONFIG_FILE = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), config_file_path)
        exec(open(CONFIG_FILE).read())
        msg = "config file '{}' found and successfully parsed".format(
            CONFIG_FILE)
        util.logging.log_info(opts.log_prefix, "parse_config", msg)
    except FileNotFoundError:
        msg = "no '{}' file found. Use defaults.".format(CONFIG_FILE)
        util.logging.log_warning(opts.log_prefix, "parse_config", msg)
    except Exception as e:
        msg = "failed to parse config file '{}'. REASON: {}. ABORT".format(
            CONFIG_FILE, str(e))
        util.logging.log_error(opts.log_prefix, "parse_config", msg)
        sys.exit(1)


def print_config_defaults():
    gpufort_python_dir = os.path.dirname(os.path.realpath(__file__))
    options_files = [
      "opts.py",
      "gpufort/fort2x/namespacegen/opts.py",
      "gpufort/fort2x/hip/opts.py",
      "gpufort/fort2x/opts.py",
      "gpufort/fort2x/gpufort_sources/opts.py",
      "gpufort/indexer/opts.py",
      "gpufort/linemapper/opts.py",
      "gpufort/scanner/opts.py",
      "gpufort/translator/opts.py",
      "gpufort/util/logging/opts.py",
    ]
    print("\nConfigurable GPUFORT options (default values):")
    for options_file in options_files:
        prefix = (options_file.replace("/",
                                       ".").replace("gpufort.",
                                                    "").replace(".py", ""))
        print("\n---- " + prefix + " -----------------------------")
        with open(os.path.join(gpufort_python_dir, options_file)) as f:
            for line in f.readlines():
                if (len(line.strip()) and
                        "# SPDX-License-Identifier:" not in line and
                        "# Copyright (c)" not in line):
                    if line[0] not in [" ", "#", "}", "]"] and "=" in line:
                        parts = line.split("=")
                        key = (prefix + "." + parts[0].rstrip()).ljust(36)
                        value = parts[1].lstrip()
                        print(key + "= " + value, end="")
                    else:
                        print(line, end="")
        print("")


def populate_cl_arg_parser(parser,for_converter=True):
    """Populate the argparse command line parser with options.
    :param bool for_converter: Include options for converter (True) or compiler (False), defaults to True
    """
    # General options
    if for_converter:
        parser.add_argument("input",
                            help="The input file.",
                            type=str,
                            nargs="?",
                            default=None)
        parser.add_argument("--output",
                            help="Name/path for the modified Fortran file.",
                            type=str,
                            dest="output",
                            nargs="?",
                            default=None)
        parser.add_argument("--output-cpp",
                            help=textwrap.dedent("""\
                                  Name/path for the main C++ file if any C++ code is generated.
                                  Per module C++ files are written into the main C++ file's
                                  parent directory.
                                  If not specified, all C++ files are written
                                  to the same directory as the modified Fortran file."""),
                            dest="output_cpp",
                            type=str,
                            nargs="?",
                            default=None)
    parser.add_argument(
        "--working-dir",
        dest="working_dir",
        default=os.getcwd(),
        type=str,
        help="Set working directory.",
    ) # shadow arg
    parser.add_argument(
        "--search-dirs",
        dest="search_dirs",
        help=
        "Module search dir. Alternative -I<path> can be used (multiple times).",
        nargs="*",
        required=False,
        default=[],
        type=str,
    )
    if for_converter:
        parser.add_argument(
            "--wrap",
            dest="wrap_in_ifdef",
            const="__GPUFORT",
            default=linemapper.opts.line_grouping_ifdef_macro,
            action="store_const",
            help=textwrap.dedent("""\
                 Wrap original and modified code as follows:
                 ```#ifdef <macro_name>\n<modified_code>\n#else\n<original_code>\n#endif\n
                 If this option is set but no macro is specified, the default '__GPUFORT' is used.
                 If this option is not set, the respective config value is used."""),
        )
        parser.add_argument(
            "--dest",
            dest="destination_dialect",
            default=None,
            type=str,
            help="One of: {}".format(", ".join(
                scanner.tree.supported_destination_dialects)),
        )
    if for_converter:
        parser.add_argument(
            "--gpufort-fc",
            dest="fortran_compiler",
            default="hipfc",
            type=str,
            help=
            "Compiler for Fortran code, defaults to HIPFORT's gfortran-based 'hipfc'. Use '--gpufort-fcflags <flags>' to set compilation flags and '--gpufort-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and C++ compiler and their respective options must be specified after all other GPUFORT options.",
        )
        parser.add_argument(
            "--gpufort-cc",
            dest="cpp_compiler",
            default="hipcc",
            type=str,
            help=
            "Compiler for C++ code, defaults to 'hipcc'. Use '--gpufort-cflags <flags>' to set compilation flags and '--gpufort-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and C++ compiler and their respective options must be specified after all other GPUFORT options.",
        )
    parser.add_argument(
        "--no-gpufort-default-flags",
        dest="use_default_flags",
        action="store_false",
        help=
        "Do not use any GPUFORT default flags but instead provide all flags explicitly via the '--gpufort-(fcflags|cflags|ldflags)' flags.",
    )
    parser.add_argument(
        "--print-gfortran-config",
        dest="print_gfortran_config",
        action="store_true",
        help=
        "Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--print-cpp-config",
        dest="print_cpp_config",
        action="store_true",
        help=
        "Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--print-ldflags",
        dest="print_ldflags",
        action="store_true",
        help=
        "Print linker flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--print-acc-ldflags",
        dest="print_ldflags_gpufort_rt",
        action="store_true",
        help=
        "Print GPUFORT accelerator runtime linker flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--print-path",
        dest="print_path",
        action="store_true",
        help="Print path to the GPUFORT root directory.",
    )
    # Frontends
    group_frontends = parser.add_argument_group("Frontends")
    group_frontends.add_argument(
        "--no-modern-fortran",
        dest="modern_fortran",
        action="store_false",
        help="Translate legacy Fortran.",
    )
    group_frontends.add_argument(
        "--cuda",
        dest="cuda",
        action="store_true",
        help="Translate CUDA Fortran statements.",
    )
    group_frontends.add_argument(
        "--no-cuda",
        dest="cuda",
        action="store_false",
        help="Do not translate CUDA Fortran statements.",
    )
    group_frontends.add_argument(
        "--cublas-v2",
        dest="cublasV2",
        action="store_true",
        help=
        "Assume cublas v2 function signatures that use a handle. Overrides config value.",
    )
    group_frontends.add_argument(
        "--openacc",
        dest="openacc",
        action="store_true",
        help="Translate OpenACC statements.",
    )
    group_frontends.add_argument(
        "--no-openacc",
        dest="openacc",
        action="store_false",
        help="Do not translate OpenACC statements.",
    )
    group_frontends.add_argument(
        "--no-translate-compute-constructs",
        dest="translate_compute_constructs",
        action="store_false",
        help="Do not translate CUDA Fortran/OpenACC/... compute constructs.",
    )
    group_frontends.add_argument(
        "--no-translate-other-directives",
        dest="translate_other_directives",
        action="store_false",
        help="Do not translate CUDA Fortran/OpenACC/... directives that are not compute constructs."
    )
    # config options: shadow arguments that are actually taken care of by raw argument parsing
    group_config = parser.add_argument_group("Config file")
    group_config.add_argument(
        "--print-config-defaults",
        dest="print_config_defaults",
        action="store_true",
        help="Print config defaults. " +
        "Config values can be overriden by providing a config file. A number of config values can be overwritten via this CLI.",
    )
    group_config.add_argument(
        "--config-file",
        default=None,
        type=argparse.FileType("r"),
        dest="config_file",
        help="Provide a config file.",
    )
    parser.add_argument(
        "--only-create-mod-files",
        dest="only_create_gpufort_module_files",
        action="store_true",
        help="Only create GPUFORT module files. Do not translate input file and generate C++ files.",
    )
    parser.add_argument(
        "--skip-create-mod-files",
        dest="skip_create_gpufort_module_files",
        action="store_true",
        help=
        "Skip creating GPUFORT modules, e.g. if they already exist. Mutually exclusive with '--only-create-mod-files' option.",
    )
    parser.add_argument(
        "--touch-cpp-files",
        dest="touch_cpp_file_per_module",
        action="store_true",
        help="""Per module found in the index derived from the current file emit an empty C++ file 
if it does not exist yet. This is realized via a touch action, i.e. the file is not overwritten if it already exists.
This step is skipped if no GPUFORT module files are created.
        """,
    )

    # fort2x.hip.codegen
    group_fort2x = parser.add_argument_group(
        "All Fortran-to-X backends (HIP,...)")
    if for_converter:
        group_fort2x.add_argument(
            "--only-modify-source",
            dest="only_modify_translation_source",
            action="store_true",
            help="Only modify Fortran code; do not generate kernels [default: False].",
        )
        group_fort2x.add_argument(
            "--only-emit-kernels-and-launchers",
            dest="only_emit_kernels_and_launchers",
            action="store_true",
            help=
            "Only emit kernels and kernel launchers; do not modify Fortran code [default: False].",
        )
        group_fort2x.add_argument(
            "--only-emit-kernels",
            dest="only_emit_kernels",
            action="store_true",
            help=
            "Only emit kernels; do not emit kernel launchers and do not modify Fortran code [default: False].",
        )
    group_fort2x_hip = parser.add_argument_group("Fortran-to-HIP")
    group_fort2x_hip.add_argument(
        "--emit-launcher-interfaces",
        dest="emit_launcher_interfaces",
        action="store_true",
        help=
        "Emit explicit Fortran interfaces to the C++ kernel launchers  [default: (default) config value].",
    )
    group_fort2x_hip.add_argument(
        "--emit-cpu-impl",
        dest="emit_cpu_implementation",
        action="store_true",
        help=
        "Per detected loop kernel, also extract the CPU implementation  [default: (default) config value].",
    )
    group_fort2x_hip.add_argument(
        "--emit-debug-code",
        dest="emit_debug_code",
        action="store_true",
        help=
        "Generate debug code into the kernel launchers that allows to print kernel arguments, launch parameters, input/output array norms and elements, or to synchronize a kernel [default: (default) config value].",
    )
    group_fort2x_hip.add_argument(
        "--emit-interop-types",
        dest="emit_interop_types",
        action="store_true",
        help=
        "Generate interoperable types from derived types found in the file.",
    )
    group_fort2x_hip.add_argument(
        "--resolve-parameters-via-fc",
        dest="resolve_parameters_via_fc",
        action="store_true",
        help=
        "Resolve parameter values via the Fortran compiler specified via '--gpufort-fc' switch.",
    )
    group_fort2x_hip.add_argument(
        "--assume-used-mod-files-exist",
        dest="assume_used_mod_files_exist",
        action="store_true",
        help=
        "When resolving parameter values via the Fortran compiler specified via"+\
        "'--gpufort-fc', assume that all used modules have been compiled already"+\
        "and a *.mod file exists. WARNING: Only use if code in the input file"+\
        "does not depend on a module within the input file.",
    )

    # developer options
    group_developer = parser.add_argument_group(
        "Developer options (logging, profiling, ...)")
    group_developer.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        action="store_true",
        help="Print all log messages to error output stream too.",
    )
    group_developer.add_argument(
        "--log-level",
        dest="log_level",
        required=False,
        type=str,
        default="",
        help="Set log level. Overrides config value.",
    )
    group_developer.add_argument(
        "--log-filter",
        dest="log_filter",
        required=False,
        type=str,
        default=None,
        help="Filter the log output according to a regular expression.",
    )
    group_developer.add_argument(
        "--log-traceback",
        dest="log_traceback",
        required=False,
        action="store_true",
        help=
        "Append gpufort traceback information to the log when encountering warning/error.",
    )
    group_developer.add_argument(
        "--dump-linemaps",
        dest="dump_linemaps",
        required=False,
        action="store_true",
        help=
        "Write the lines-to-statements mappings to disk pre & post applying code transformations.",
    )
    group_developer.add_argument(
        "--prof",
        dest="profiling_enable",
        required=False,
        action="store_true",
        help="Profile gpufort.",
    )
    group_developer.add_argument(
        "--prof-num-functions",
        dest="profiling_num_functions",
        required=False,
        type=int,
        default=50,
        help=
        "The number of python functions to include into the summary [default=50].",
    )
    group_developer.add_argument(
        "--create-gpufort-headers",
        dest="create_gpufort_headers",
        action="store_true",
        help="Generate the GPUFORT header files.",
    )
    group_developer.add_argument(
        "--create-gpufort-sources",
        dest="create_gpufort_sources",
        action="store_true",
        help="Generate the GPUFORT source files.",
    )

    parser.set_defaults(
        print_config_defaults=False,
        dump_linemaps=False,
        modern_fortran=True,
        cuda=True,
        openacc=True,
        translate_compute_constructs=True,
        translate_other_directives=True,
        cublasV2=False,
        only_emit_kernels_and_launchers=False,
        only_emit_kernels=False,
        only_modify_translation_source=False,
        emit_cpu_implementation=False,
        emit_debug_code=False,
        emit_interop_types=False,
        emit_launcher_interfaces=False,
        resolve_parameters_via_fc=False,
        assume_used_mod_files_exist=False,
        create_gpufort_headers=False,
        create_gpufort_sources=False,
        print_gfortran_config=False,
        print_cpp_config=False,
        print_ldflags=False,
        print_ldflags_gpufort_rt=False,
        only_create_gpufort_module_files=False,
        skip_create_gpufort_module_files=False,
        verbose=False,
        log_traceback=False,
        profiling_enable=False,
        use_default_flags=True,
    )

def get_basic_cflags():
    cflags = []
    hip_platform = os.environ.get("HIP_PLATFORM", "amd")
    if linemapper.opts.line_grouping_ifdef_macro != None:
        cflags.append("-D" + linemapper.opts.line_grouping_ifdef_macro)
    cflags.append("-I" + os.path.join(__GPUFORT_ROOT_DIR, "include"))
    cflags.append("-I" + \
        os.path.join(__GPUFORT_ROOT_DIR, "include", hip_platform))
    return cflags

def get_hipcc_cflags():
    return get_basic_cflags() + ["-fPIC","-std=c++14"]

def get_gfortran_cflags():
    return ["-cpp","-ffree-line-length-none"]

def get_basic_ldflags(with_runtime):
    ldflags = []
    hip_platform = os.environ.get("HIP_PLATFORM", "amd")
    ldflags.append("".join(["-L",os.path.join(__GPUFORT_ROOT_DIR, "lib")]))
    if with_runtime:
        ldflags.append("".join(["-lgpufortrt_",hip_platform]))
    ldflags.append("".join(["-lgpufort_",hip_platform]))
    return ldflags

def parse_cl_args(parser,for_converter=True):
    """Parse command line arguments after all changes and argument transformations by the config file have been applied.
    :param bool for_converter: Include options for converter (True) or compiler (False), defaults to True
    """
    args, unknown_args = parser.parse_known_args()

    # Simple output commands
    if args.print_path:
        print(__GPUFORT_ROOT_DIR, file=sys.stdout)
        sys.exit()
    if args.print_cpp_config:
        print(" ".join(get_hipcc_cflags()), file=sys.stdout)
        sys.exit()
    if args.print_gfortran_config:
        print(" ".join(get_gfortran_cflags()+get_basic_cflags()), file=sys.stdout)
        sys.exit()
    if args.print_ldflags:
        print(" ".join(get_basic_ldflags(False)), file=sys.stdout)
        sys.exit()
    if args.print_ldflags_gpufort_rt:
        print(" ".join(get_basic_ldflags(True)), file=sys.stdout)
        sys.exit()
    call_exit = False
    if args.create_gpufort_headers:
        fort2x.gpufort_sources.generate_gpufort_headers(os.getcwd())
        call_exit = True
    if args.create_gpufort_sources:
        fort2x.gpufort_sources.generate_gpufort_sources(os.getcwd())
        call_exit = True
    if call_exit:
        sys.exit(0)
    if args.print_config_defaults:
        print_config_defaults()
        sys.exit(0)
    # VALIDATION
    if args.only_create_gpufort_module_files and args.skip_create_gpufort_module_files:
        msg = "switches '--only-create-mod-files' and '--skip-generate-mod-files' are mutually exclusive."
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    # mutually exclusive arguments
    if (int(args.only_emit_kernels_and_launchers) + int(args.only_emit_kernels)
            + int(args.only_modify_translation_source)) > 1:
        msg = "switches '--only-emit-kernels', '--only-emit-kernels-and-launchers', and 'only-modify-fortran-code' are mutually exclusive."
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    # unknown options
    if len([u for u in unknown_args if u.replace("--gpufort-","") not in fortran_and_cpp_compiler_options]):
        msg = "unknown arguments (may be used by registered actions): {}".format(
            " ".join(unknown_args))
        print("WARNING: " + msg, file=sys.stderr)
    # check input and output paths
    if for_converter:
        if args.input is None:
            msg = "no input file"
            print("ERROR: " + msg, file=sys.stderr)
            sys.exit(2)
        if not os.path.isabs(args.input):
            args.input = os.path.join(args.working_dir, args.input)
        if args.output != None and not os.path.isabs(args.output):
            args.output = os.path.join(args.working_dir, args.output)
        if args.output_cpp != None and not os.path.isabs(args.output_cpp):
            if args.output != None:
                args.output_cpp = os.path.join(os.path.dirname(args.output), args.output_cpp)
            else:
                args.output_cpp = os.path.join(args.working_dir, args.output_cpp)
        if not os.path.exists(args.input):
            msg = "input file '{}' cannot be found".format(args.input)
            print("ERROR: " + msg, file=sys.stderr)
            sys.exit(2)
        if args.output != None and not os.path.exists(os.path.dirname(args.output)):
            msg = "output directory '{}' does not exist".format(os.path.dirname(args.output))
            print("ERROR: " + msg, file=sys.stderr)
            sys.exit(2)
        if args.output_cpp != None and not os.path.exists(os.path.dirname(args.output_cpp)):
            msg = "output directory for C++ files '{}' does not exist".format(os.path.dirname(args.output_cpp))
            print("ERROR: " + msg, file=sys.stderr)
            sys.exit(2)
        if args.output == None:
            args.output = "".join([args.input,opts.fortran_file_ext]) 
        if args.output_cpp == None:
            args.output_cpp = "".join([args.input,opts.cpp_file_ext]) 
    #
    return args, unknown_args


def map_args_to_opts(args,include_dirs,defines,fortran_and_cpp_compiler_options,for_converter=True):
    """Map specified command line arguments to option values.
    :note: Only overwrites the option values if a flag is actually specified.
    """
    # OVERWRITE CONFIG VALUES
    # parse file and create index in parallel
    if for_converter: 
        if args.destination_dialect != None:
            scanner.opts.destination_dialect = scanner.tree.check_destination_dialect(
                args.destination_dialect)
    # frontends
    linemapper.opts.modern_fortran = args.modern_fortran
    indexer.opts.modern_fortran = args.modern_fortran
    scanner.opts.modern_fortran = args.modern_fortran
    translator.opts.modern_fortran = args.modern_fortran
    #
    scanner.opts.translate_compute_constructs = args.translate_compute_constructs
    scanner.opts.translate_other_directives = args.translate_other_directives
    #
    linemapper.opts.cuda_fortran = args.cuda
    indexer.opts.cuda_fortran = args.cuda
    if args.cuda:
        scanner.opts.source_dialects.add("cuf")
    else:
        scanner.opts.source_dialects.remove("cuf")
    #
    indexer.opts.openacc = args.openacc
    if args.openacc:
        scanner.opts.source_dialects.add("acc")
    else:
        scanner.opts.source_dialects.remove("acc")

    # only create modules files / skip module file generation
    if args.only_create_gpufort_module_files:
        opts.only_create_gpufort_module_files = True
    if args.skip_create_gpufort_module_files:
        opts.skip_create_gpufort_module_files = True
    if args.touch_cpp_file_per_module:
        opts.touch_cpp_file_per_module = True
    # only generate kernels / modify source
    if args.only_emit_kernels_and_launchers:
        opts.only_emit_kernels_and_launchers = True
    if args.only_emit_kernels:
        opts.only_emit_kernels = True
    if args.only_modify_translation_source:
        opts.only_modify_translation_source = True
    # todo: move to different location
    # set default values, add main includes and preprocessor definitions
    # to fortran and cpp parser options.
    arg_fc      = "-fc"
    arg_fcflags = "-fcflags"
    arg_cc      = "-cc"
    arg_cflags  = "-cflags"
    fcflags = fortran_and_cpp_compiler_options[arg_fcflags]
    if args.use_default_flags:
        fcflags = shlex.split(fort2x.namespacegen.opts.fortran_compiler_flags) + fcflags
        fcflags  = ["-I{}".format(inc) for inc in include_dirs] + defines + fcflags
        fortran_and_cpp_compiler_options[arg_fcflags] = fcflags
    cflags = fortran_and_cpp_compiler_options[arg_cflags]
    if args.use_default_flags:
        cflags = ["-I{}".format(inc) for inc in include_dirs] + defines + cflags
        fortran_and_cpp_compiler_options[arg_cflags] = cflags
    #arg_ldflags = "-".join(["","gpufort","ldflags"])
    # fort2x.namespacegen 
    if len(fortran_and_cpp_compiler_options[arg_fc]):
        fort2x.namespacegen.opts.fortran_compiler=" ".join(fortran_and_cpp_compiler_options[arg_fc])
    if len(fortran_and_cpp_compiler_options[arg_cc]):
        fort2x.namespacegen.opts.fortran_compiler=" ".join(fortran_and_cpp_compiler_options[arg_cc])
    if len(fortran_and_cpp_compiler_options[arg_fcflags]) or args.use_default_flags:
        fort2x.namespacegen.opts.fortran_compiler_flags = " ".join(fcflags)
    if args.resolve_parameters_via_fc:
        fort2x.namespacegen.opts.resolve_all_parameters_via_compiler = True
    if args.assume_used_mod_files_exist:
        fort2x.namespacegen.opts.all_used_modules_have_been_compiled = True
    # fort2x.hip.codegen
    if opts.only_emit_kernels:
        fort2x.hip.opts.emit_cpu_launcher = False
        fort2x.hip.opts.emit_grid_launcher = False
        fort2x.hip.opts.emit_problem_size_launcher = False
    if args.emit_cpu_implementation:
        fort2x.hip.opts.emit_cpu_launcher = True
    if args.emit_launcher_interfaces:
        fort2x.hip.opts.emit_launcher_interfaces = True
    if args.emit_debug_code:
        fort2x.hip.opts.emit_debug_code = True
    if args.emit_interop_types:
        fort2x.hip.opts.emit_interop_types = True
    # wrap modified lines in ifdef
    if for_converter:
        linemapper.opts.line_grouping_ifdef_macro = args.wrap_in_ifdef
    # developer: logging
    if len(args.log_level):
        opts.log_level = args.log_level
    if args.verbose:
        util.logging.opts.verbose = True
    if args.log_traceback:
        util.logging.opts.traceback = True
    if args.log_filter != None:
        util.logging.opts.log_filter = args.log_filter
    # developer: profiling:
    if args.profiling_enable:
        opts.profiling_enable = True
        opts.profiling_output_num_functions = args.profiling_num_functions
    # developer: other
    if args.dump_linemaps:
        opts.dump_linemaps = True
    # CUDA Fortran
    if args.cublasV2:
        scanner.opts.cublas_version = 2
        translator.opts.cublas_version = 2


def init_logging(infile_path):
    """Init logging infrastructure.
    :param str infile_path: Input file path."""
    infile_path_hash = hashlib.md5(
        infile_path.encode()).hexdigest()[0:8]
    logfile_basename = "log-{}.log".format(infile_path_hash)

    log_format = opts.log_format.replace("%(filename)s", infile_path)
    log_file_path = util.logging.init_logging(logfile_basename, log_format,
                                              opts.log_level)

    msg = "input file: {0} (log id: {1})".format(infile_path,
                                                 infile_path_hash)
    util.logging.log_info(opts.log_prefix, "init_logging", msg)
    msg = "log file:   {0} (log level: {1}) ".format(log_file_path,
                                                     opts.log_level)
    util.logging.log_info(opts.log_prefix, "init_logging", msg)
    return log_file_path


def shutdown_logging(log_file_path):
    """Shutdown the logging infrastructure."""
    msg = "log file:   {0} (log level: {1}) ".format(log_file_path,
                                                     opts.log_level)
    util.logging.log_info(opts.log_prefix, "__main__", msg)
    util.logging.shutdown()

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index(linemaps, output_dir, search_dirs):
    """:return: Index data structure plus the number
             of top level entries contributed by the current file.
             (-1 if no gpufort module files are created and they are only read.)
    """
    index = []
    num_top_level_entries = -1
    if not opts.skip_create_gpufort_module_files:
        indexer.update_index_from_linemaps(linemaps, index)
        num_top_level_entries = len(index)
        if opts.touch_cpp_file_per_module:
            touch_cpp_file_per_module(index, output_dir)
        indexer.write_gpufort_module_files(index, output_dir)
    if not opts.only_create_gpufort_module_files:
        index.clear()
        indexer.load_gpufort_module_files(search_dirs, index)
    return index, num_top_level_entries

@util.logging.log_entry_and_exit(opts.log_prefix)
def touch_cpp_file_per_module(index, output_dir):
    for irecord in index:
        if irecord["kind"] == "module":
            cpp_file_name = "".join([irecord["name"],fort2x.opts.cpp_file_ext])
            cpp_file_path = os.path.join(output_dir,cpp_file_name)
            pathlib.Path(cpp_file_path).touch()
            msg = "Touch C++ file: ".ljust(40) + cpp_file_path
            util.logging.log_info(opts.log_prefix, "touch_cpp_file_per_module", msg)

@util.logging.log_entry_and_exit(opts.log_prefix)
def translate_source(infile_path, outfile_path, stree, linemaps, index, preamble):
    # post process
    scanner.tree.postprocess(
        stree,
        index) # , fortran_module_suffix=fort2x.opts.fortran_module_suffix)

    # transform statements to 'linemaps'
    def transform_(stnode):
        stnode.transform_statements(index)
        for child in stnode.children:
            transform_(child)

    transform_(stree)

    # write the file
    linemapper.write_modified_file(outfile_path, infile_path, linemaps,
                                   preamble)

    # prettify the file
    # if opts.fortran_prettifier != None
    #    util.file.prettify_f_file(outfile_path)
    msg = "created translated input file: ".ljust(40) + outfile_path
    util.logging.log_info(opts.log_prefix, "translate_source", msg)

    return outfile_path


def run(infile_path,outfile_path,outfile_cpp_path,preproc_options):
    cpp_file_paths = []

    """Converter runner."""
    if opts.profiling_enable:
        profiler = cProfile.Profile()
        profiler.enable()

    linemaps = linemapper.read_file(infile_path,
            preproc_options=preproc_options,
            include_dirs=opts.include_dirs)
    if opts.dump_linemaps:
        util.logging.log_info(opts.log_prefix, "__main__",
                              "dump linemaps (before translation)")
        linemapper.dump_linemaps(linemaps,
                                 infile_path + "-linemaps-pre.json")
    index, num_local_top_level_index_entries = create_index(linemaps,os.path.dirname(infile_path),opts.include_dirs)
    if not opts.only_create_gpufort_module_files:
        stree, _, __ = scanner.parse_file(file_linemaps=linemaps,
                                          file_path=infile_path,
                                          index=index)
        if "hip" in scanner.opts.destination_dialect:
            codegen = fort2x.hip.hipcodegen.HipCodeGenerator(stree, index)#, num_top_level_index_entries)
            codegen.run()
            cpp_file_paths += codegen.write_cpp_files(outfile_cpp_path)

        if not (opts.only_emit_kernels or
                opts.only_emit_kernels_and_launchers):
            translate_source(infile_path, outfile_path, stree, linemaps, index,
                             opts.fortran_file_preamble)
    #
    if opts.profiling_enable:
        profiler.disable()
        s = io.StringIO()
        sortby = "cumulative"
        stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        stats.print_stats(opts.profiling_output_num_functions)
        print(s.getvalue())

    if opts.dump_linemaps:
        util.logging.log_info(opts.log_prefix, "__main__",
                              "dump linemaps (after translation)")
        linemapper.dump_linemaps(linemaps,
                                 infile_path + "-linemaps-post.json")
    return cpp_file_paths

def run_checked(*args,**kwargs):
    try:
        run(*args,**kwargs)
    except (ValueError,
            IOError,
            FileNotFoundError,
            util.error.SyntaxError,
            util.error.SemanticError,
            util.error.LimitationError,
            util.error.LookupError) as e:
        msg = str(e)
        print("ERROR: "+msg)
        util.logging.log_exception(opts.log_prefix, "run_checked", msg)
    except Exception as e:
        raise e

def set_include_dirs(working_dir, include_dirs):
    """Update opts.include_dirs from all sources."""
    opts.include_dirs += include_dirs
    one_or_more_search_dirs_not_found = False
    for i, directory in enumerate(opts.include_dirs):
        if not os.path.isabs(directory):
            opts.include_dirs[i] = os.path.join(working_dir, directory)
        if not os.path.exists(opts.include_dirs[i]):
            msg = "search directory '{}' cannot be found".format(
                opts.include_dirs[i])
            util.logging.log_error(opts.log_prefix,"set_include_dirs",msg)
            one_or_more_search_dirs_not_found = True
    opts.include_dirs.insert(0,working_dir)
    if one_or_more_search_dirs_not_found:
        sys.exit(2)

if __name__ == "__main__":
    # parse config
    config_file_path, include_dirs, \
    defines, fortran_and_cpp_compiler_options = parse_raw_cl_args()
    if config_file_path != None:
        parse_config(config_file_path)
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="S2S translation tool for CUDA Fortran and Fortran+X")
    populate_cl_arg_parser(parser) 
    args, unknown_args = parse_cl_args(parser)
    if len(opts.post_cli_actions):
        msg = "run registered actions"
        util.logging.log_info(opts.log_prefix,"__main__",msg, verbose=False)
        for action in opts.post_cli_actions:
            if callable(action):
                action(args, unknown_args)
    set_include_dirs(args.working_dir, args.search_dirs + include_dirs)
    map_args_to_opts(args,opts.include_dirs,defines,fortran_and_cpp_compiler_options)

    infile_path = os.path.abspath(args.input) 
    outfile_path = os.path.abspath(args.output) 
    outfile_cpp_path = os.path.abspath(args.output_cpp) 
    log_file_path = init_logging(infile_path)

    run_checked(infile_path,outfile_path,outfile_cpp_path,defines)

    shutdown_logging(log_file_path)
