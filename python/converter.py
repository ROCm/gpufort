#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import argparse
import hashlib
import cProfile
import pstats
import io

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

# arg for kernel generator
# array is split into multiple args


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
    return config_file_path, include_dirs, defines


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
        "gpufort/fort2x/opts.py",
        "gpufort/fort2x/hip/opts.py",
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
    parser.add_argument(
        "--wrap",
        dest="wrap_in_ifdef",
        const="_GPUFORT",
        default=linemapper.opts.line_grouping_ifdef_macro,
        action="store_const",
        help="Wrap original and modified code as follows:\n"\
             + "```#ifdef <macro_name>\n<modified_code>\n#else\n<original_code>\n#endif\n```"\
             +"If this option is set but no macro is specified, the default '_GPUFORT' is used.",
    )
    if for_converter:
        parser.add_argument(
            "--dest",
            dest="destination_dialect",
            default=None,
            type=str,
            help="One of: {}".format(", ".join(
                scanner.tree.supported_destination_dialects)),
        )
    parser.add_argument(
        "--gfortran_config",
        dest="print_gfortran_config",
        action="store_true",
        help=
        "Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--cpp_config",
        dest="print_cpp_config",
        action="store_true",
        help=
        "Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--ldflags",
        dest="print_ldflags",
        action="store_true",
        help=
        "Print linker flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--ldflags-gpufort-rt",
        dest="print_ldflags_gpufort_rt",
        action="store_true",
        help=
        "Print GPUFORT OpenACC runtime linker flags; output is influenced by HIP_PLATFORM environment variable.",
    )
    parser.add_argument(
        "--path",
        dest="print_path",
        action="store_true",
        help="Print path to the GPUFORT root directory.",
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
        help="Only create GPUFORT modules files. No other output is created.",
    )
    parser.add_argument(
        "--skip-create-mod-files",
        dest="skip_create_gpufort_module_files",
        action="store_true",
        help=
        "Skip creating GPUFORT modules, e.g. if they already exist. Mutually exclusive with '-c' option.",
    )

    # fort2x.hip.codegen
    group_fort2x = parser.add_argument_group(
        "All Fortran-to-X backends (HIP,...)")
    if for_converter:
        group_fort2x.add_argument(
            "--only-modify-source",
            dest="only_modify_translation_source",
            action="store_true",
            help="Only modify host code; do not generate kernels [default: False].",
        )
        group_fort2x.add_argument(
            "--only-emit-kernels-and-launchers",
            dest="only_emit_kernels_and_launchers",
            action="store_true",
            help=
            "Only emit kernels and kernel launchers; do not modify host code [default: False].",
        )
        group_fort2x.add_argument(
            "--only-emit-kernels",
            dest="only_emit_kernels",
            action="store_true",
            help=
            "Only emit kernels; do not emit kernel launchers and do not modify host code [default: False].",
        )
    group_fort2x_hip = parser.add_argument_group("Fortran-to-HIP")
    if for_converter:
        group_fort2x_hip.add_argument(
            "--emit-launcher-interfaces",
            dest="emit_launcher_interfaces",
            action="store_true",
            help=
            "Emit explicit Fortran interfaces to the C++ kernel launchers  [default: (default) config value].",
        )
    group_fort2x_hip.add_argument(
        "--emit-grid-launcher",
        dest="emit_cpu_implementation",
        action="store_true",
        help=
        "Per detected loop kernel, also emit a grid launcher [default: (default) config value].",
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

    # CUDA Fortran
    group_cuf = parser.add_argument_group("CUDA Fortran input")
    group_cuf.add_argument(
        "--cublas-v2",
        dest="cublasV2",
        action="store_true",
        help=
        "Assume cublas v2 function signatures that use a handle. Overrides config value.",
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
        cublasV2=False,
        only_emit_kernels_and_launchers=False,
        only_emit_kernels=False,
        only_modify_translation_source=False,
        emit_cpu_implementation=False,
        emit_debug_code=False,
        emit_launcher_interfaces=False,
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
    return ["-fPIC"]

def get_gfortran_cflags():
    return ["-cpp","-std=f2008","-ffree-line-length-none"]

def get_basic_ldflags(with_runtime):
    ldflags = []
    hip_platform = os.environ.get("HIP_PLATFORM", "amd")
    if with_runtime:
        ldflags.append("".join([
            " -L",
            os.path.join(__GPUFORT_ROOT_DIR, "lib"), " -lgpufort_acc_",
            hip_platform, " -lgpufort_", hip_platform
            ]))
    else:
        ldflags.append("".join([
            " -L",
            os.path.join(__GPUFORT_ROOT_DIR, "lib"), " -lgpufort_",
            hip_platform
            ]))
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
        print(" ".join(get_basic_cflags()), file=sys.stdout)
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
        msg = "switches '--only-emit-kernels', '--only-emit-kernels-and-launchers', and 'only-modify-host-code' are mutually exclusive."
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    # unknown options
    if len(unknown_args):
        msg = "unknown arguments (may be used by registered actions): {}".format(
            " ".join(unknown_args))
        print("WARNING: " + msg, file=sys.stderr)
    # check if input is set
    if args.input is None:
        msg = "no input file"
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    if args.input[0] != "/":
        args.input = args.working_dir + "/" + args.input
    if not os.path.exists(args.input):
        msg = "input file '{}' cannot be found".format(args.input)
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(2)
    #
    return args, unknown_args


def map_args_to_opts(args):
    # OVERWRITE CONFIG VALUES
    # parse file and create index in parallel
    if args.destination_dialect != None:
        scanner.opts.destination_dialect = scanner.tree.check_destination_dialect(
            args.destination_dialect)
    # only create modules files / skip module file generation
    if args.only_create_gpufort_module_files:
        opts.only_create_gpufort_module_files = True
    if args.skip_create_gpufort_module_files:
        opts.skip_create_gpufort_module_files = True
    # fort2x.hip.codegen
    # only generate kernels / modify source
    if args.only_emit_kernels_and_launchers:
        opts.only_emit_kernels_and_launchers = True
    if args.only_emit_kernels:
        opts.only_emit_kernels = True
    if args.only_modify_translation_source:
        opts.only_modify_translation_source = True
    # configure fort2x.hip.codegen
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
    # wrap modified lines in ifdef
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

def create_index(search_dirs, options, file_path, linemaps=None):
    util.logging.log_enter_function(
        opts.log_prefix,
        "create_index",
        {
            "file_path": file_path,
            "options": " ".join(options),
            "search_dirs": " ".join(search_dirs),
        },
    )

    options_as_str = " ".join(options)

    index = []
    if not opts.skip_create_gpufort_module_files:
        if linemaps != None:
            indexer.update_index_from_linemaps(linemaps, index)
        else:
            indexer.scan_file(file_path, options_as_str, index)
        output_dir = os.path.dirname(file_path)
        indexer.write_gpufort_module_files(index, output_dir)
    index.clear()
    indexer.load_gpufort_module_files(search_dirs, index)

    util.logging.log_leave_function(opts.log_prefix, "create_index")
    return index

@util.logging.log_entry_and_exit(opts.log_prefix)
def translate_source(infile_path, stree, linemaps, index, preamble):
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
    outfile_path = infile_path + opts.fortran_file_ext
    linemapper.write_modified_file(outfile_path, infile_path, linemaps,
                                   preamble)

    # prettify the file
    # if opts.fortran_prettifier != None
    #    util.file.prettify_f_file(outfile_path)
    msg = "created hipified input file: ".ljust(40) + outfile_path
    util.logging.log_info(opts.log_prefix, "translate_source", msg)

    return outfile_path


def run(infile_path,preproc_options):
    """Converter runner."""
    if opts.profiling_enable:
        profiler = cProfile.Profile()
        profiler.enable()

    linemaps = linemapper.read_file(infile_path, preproc_options)
    if opts.dump_linemaps:
        util.logging.log_info(opts.log_prefix, "__main__",
                              "dump linemaps (before translation)")
        linemapper.dump_linemaps(linemaps,
                                 infile_path + "-linemaps-pre.json")

    index = create_index(opts.include_dirs, preproc_options, infile_path,
                         linemaps)
    fortran_file_paths = []
    cpp_file_paths     = []
    if not opts.only_create_gpufort_module_files:
        stree, _, __ = scanner.parse_file(file_linemaps=linemaps,
                                          file_path=infile_path,
                                          index=index)
        if "hip" in scanner.opts.destination_dialect:
            codegen = fort2x.hip.hipcodegen.HipCodeGenerator(stree, index)
            codegen.run()
            cpp_file_paths += codegen.write_cpp_files("".join(
                [infile_path, opts.cpp_file_ext]))

        if not (opts.only_emit_kernels or
                opts.only_emit_kernels_and_launchers):
            outfile_path = translate_source(infile_path, stree, linemaps, index,
                           opts.fortran_file_preamble)
            fortran_file_paths.append(outfile_path)
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
    return fortran_file_paths, cpp_file_paths

def run_checked(*args,**kwargs):
    try:
        run(*args,**kwargs)
    except IOError as e:
        print("ERROR:"+str(e))
    except util.error.SyntaxError as e:
        print("ERROR:"+str(e))
    except util.error.LimitationError as e:
        print("ERROR:"+str(e))
    except util.error.LookupError as e:
        print("ERROR:"+str(e))
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
            util.logging.log_error(msg, verbose=False)
            one_or_more_search_dirs_not_found = True
    opts.include_dirs.append(working_dir)
    if one_or_more_search_dirs_not_found:
        sys.exit(2)

if __name__ == "__main__":
    # parse config
    config_file_path, include_dirs, defines = parse_raw_cl_args()
    if config_file_path != None:
        parse_config(config_file_path)
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="S2S translation tool for CUDA Fortran and Fortran+X")
    populate_cl_arg_parser(parser) 
    args, unknown_args = parse_cl_args(parser)
    if len(opts.post_cli_actions):
        msg = "run registered actions"
        util.logging.log_info(msg, verbose=False)
        for action in opts.post_cli_actions:
            if callable(action):
                action(args, unknown_args)
    map_args_to_opts(args)
    set_include_dirs(args.working_dir, args.search_dirs + include_dirs)

    infile_path = os.path.abspath(args.input) 
    log_file_path   = init_logging(infile_path)

    run_checked(infile_path,defines)

    shutdown_logging(log_file_path)
