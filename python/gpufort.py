#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os, sys
import argparse
import hashlib
import cProfile,pstats,io

# local imports
import addtoplevelpath
import utils.logging
import utils.fileutils
import scanner.scanner as scanner
import indexer.indexer as indexer
import linemapper.linemapper as linemapper
import translator.translator as translator
import fort2x.hip.fort2hip as fort2hip
#import fort2x.hip.fort2hip_legacy as fort2hip

__GPUFORT_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
__GPUFORT_ROOT_DIR   = os.path.abspath(os.path.join(__GPUFORT_PYTHON_DIR,".."))
exec(open(os.path.join(__GPUFORT_PYTHON_DIR, "gpufort_options.py.in")).read())

# arg for kernel generator
# array is split into multiple args

def create_index(search_dirs,options,filepath,linemaps=None):
    global LOG_PREFIX
    global SKIP_CREATE_GPUFORT_MODULE_FILES
   
    utils.logging.log_enter_function(LOG_PREFIX,"create_index",\
      {"filepath":filepath,"options":" ".join(options),\
       "search_dirs":" ".join(search_dirs)})

    options_as_str = " ".join(options)
    
    index = []
    if not SKIP_CREATE_GPUFORT_MODULE_FILES:
        if linemaps != None:
            indexer.update_index_from_linemaps(linemaps,index)
        else:
            indexer.scan_file(filepath,options_as_str,index)
        output_dir = os.path.dirname(filepath)
        indexer.write_gpufort_module_files(index,output_dir)
    index.clear()
    indexer.load_gpufort_module_files(search_dirs,index)
    
    utils.logging.log_leave_function(LOG_PREFIX,"create_index")
    return index

def _intrnl_translate_source(infilepath,stree,linemaps,index,preamble):
    global LOG_PREFIX
    global MODIFIED_FILE_EXT
    global PRETTIFY_MODIFIED_TRANSLATION_SOURCE
    
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_translate_source",{"infilepath":infilepath})
    
    # post process
    scanner.postprocess(stree,index,fort2x.hip.FORTRAN_MODULE_SUFFIX)
    
    # transform statements; to 'linemaps'
    def transform_(stnode):
        stnode.transform_statements(index)
        for child in stnode.children:
            transform_(child)
    transform_(stree)

    # write the file
    outfilepath = infilepath + MODIFIED_FILE_EXT
    linemapper.write_modified_file(outfilepath,infilepath,linemaps,preamble)

    # prettify the file
    if PRETTIFY_MODIFIED_TRANSLATION_SOURCE:
        utils.fileutils.prettify_f_file(outfilepath)
    msg = "created hipified input file: ".ljust(40) + outfilepath
    utils.logging.log_info(LOG_PREFIX,"_intrnl_translate_source",msg)
    
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_translate_source")

def parse_raw_command_line_arguments():
    """
    Parse command line arguments before using argparse.
    Allows the loaded config file to modify
    the argparse switches and help information.
    Further transform some arguments.
    """
    config_filepath  = None
    working_dir_path = os.getcwd()
    include_dirs     = []
    defines          = []
    options          = sys.argv[1:]
    for i,opt in enumerate(list(options)):
        if opt == "--working-dir":
            if i+1 < len(options):
                working_dir_path = options[i+1]
        elif opt == "--config-file":
            if i+1 < len(options):
                config_filepath = options[i+1]
        elif opt.startswith("-I"):
            include_dirs.append(opt[2:])
            sys.argv.remove(opt)
        elif opt.startswith("-D"):
            defines.append(opt)
            sys.argv.remove(opt)
    if not os.path.exists(working_dir_path):
        msg = "working directory '{}' cannot be found".format(working_dir_path)
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    if config_filepath != None: 
        if config_filepath[0] != "/":
          config_filepath = working_dir_path + "/" + config_filepath 
        if not os.path.exists(config_filepath):
            msg = "config file '{}' cannot be found".format(config_filepath)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(2)
    return config_filepath, include_dirs, defines 

def parse_config(config_filepath):
    """
    Load user-supplied config file.
    """
    gpufort_python_dir=os.path.dirname(os.path.realpath(__file__))
    options_file = "gpufort_options.py.in"
    prolog = ""
    for line in open(gpufort_python_dir+"/"+options_file,"r").readlines():
        if line and line[0].isalpha() and line[0].isupper():
            prolog += "global "+line.split("=")[0].rstrip(" \t") + "\n"
    #print(prolog)
    prolog.rstrip("\n")
    try:
        if config_filepath.strip()[0]=="/":
            CONFIG_FILE = config_filepath
        else:
            CONFIG_FILE=os.path.dirname(os.path.realpath(__file__))+"/"+config_filepath
        exec(prolog + "\n" + open(CONFIG_FILE).read())
        msg = "config file '{}' found and successfully parsed".format(CONFIG_FILE)
        utils.logging.log_info(LOG_PREFIX,"parse_config",msg)
    except FileNotFoundError:
        msg = "no '{}' file found. Use defaults.".format(CONFIG_FILE)
        utils.logging.log_warning(LOG_PREFIX,"parse_config",msg)
    except Exception as e:
        msg = "failed to parse config file '{}'. REASON: {} (NOTE: num prolog lines: {}). ABORT".format(CONFIG_FILE,str(e),len(prolog.split("\n")))
        utils.logging.log_error(LOG_PREFIX,"parse_config",msg)
        sys.exit(1)

def parse_cl_args():
    """
    Parse command line arguments after all changes and argument transformations by the config file have been applied.
    """
    global LOG_LEVEL
    global LOG_FORMAT
    global LOG_PREFIX
    global PROFILING_ENABLE
    global PROFILING_OUTPUT_NUM_FUNCTIONS
    global ONLY_CREATE_GPUFORT_MODULE_FILES
    global SKIP_CREATE_GPUFORT_MODULE_FILES
    global ONLY_MODIFY_TRANSLATION_SOURCE
    global ONLY_EMIT_KERNELS_AND_LAUNCHERS
    global ONLY_EMIT_KERNELS
    global POST_CLI_ACTIONS
    global PRETTIFY_MODIFIED_TRANSLATION_SOURCE
    global INCLUDE_DIRS
    global DUMP_LINEMAPS
    global DUMP_LINEMAPS

    # parse command line arguments
    parser = argparse.ArgumentParser(description="S2S translation tool for CUDA Fortran and Fortran+X")
    
    # General options
    parser.add_argument("input",help="The input file.",type=str,nargs="?",default=None)
    parser.add_argument("-c","--only-create-mod-files",dest="only_create_gpufort_module_files",action="store_true",help="Only create GPUFORT modules files. No other output is created.")
    parser.add_argument("-s","--skip-create-mod-files",dest="skip_create_gpufort_module_files",action="store_true",help="Skip creating GPUFORT modules, e.g. if they already exist. Mutually exclusive with '-c' option.")
    parser.add_argument("-o","--output", help="The output file. Interface module and HIP C++ implementation are named accordingly. GPUFORT module files are created too.", default=sys.stdout, required=False, type=argparse.FileType("w"))
    parser.add_argument("--working-dir",dest="working_dir",default=os.getcwd(),type=str,help="Set working directory.") # shadow arg
    parser.add_argument("-d","--search-dirs", dest="search_dirs", help="Module search dir. Alternative -I<path> can be used (multiple times).", nargs="*",  required=False, default=[], type=str)
    parser.add_argument("-w","--wrap-in-ifdef",dest="wrap_in_ifdef",action="store_true",help="Wrap converted lines into ifdef in host code.")
    parser.add_argument("-E","--dest-dialect",dest="destination_dialect",default=None,type=str,help="One of: {}".format(", ".join(scanner.SUPPORTED_DESTINATION_DIALECTS)))
    parser.add_argument("--gfortran_config",dest="print_gfortran_config",action="store_true",help="Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.")
    parser.add_argument("--cpp_config",dest="print_cpp_config",action="store_true",help="Print include and compile flags; output is influenced by HIP_PLATFORM environment variable.")
    parser.add_argument("--ldflags",dest="print_ldflags",action="store_true",help="Print linker flags; output is influenced by HIP_PLATFORM environment variable.")
    parser.add_argument("--ldflags-gpufort-rt",dest="print_ldflags_gpufort_rt",action="store_true",help="Print GPUFORT OpenACC runtime linker flags; output is influenced by HIP_PLATFORM environment variable.")
    parser.add_argument("--path",dest="print_path",action="store_true",help="Print path to the GPUFORT root directory.")
    # config options: shadow arguments that are actually taken care of by raw argument parsing
    group_config = parser.add_argument_group('Config file')
    group_config.add_argument("--print-config-defaults",dest="print_config_defaults",action="store_true",help="Print config defaults. "+\
            "Config values can be overriden by providing a config file. A number of config values can be overwritten via this CLI.")
    group_config.add_argument("--config-file",default=None,type=argparse.FileType("r"),dest="config_file",help="Provide a config file.")
    
    # fort2hip
    group_fort2x = parser.add_argument_group('All Fortran-to-X backends (HIP,...)')
    group_fort2x.add_argument("-m","--only-modify-host-code",dest="only_modify_translation_source",action="store_true",help="Only modify host code; do not generate kernels [default: False].")
    group_fort2x.add_argument("-k","--only-emit-kernels-and-launchers",dest="only_emit_kernels_and_launchers",action="store_true",help="Only emit kernels and kernel launchers; do not modify host code [default: False].")
    group_fort2x.add_argument("-K","--only-emit-kernels",dest="only_emit_kernels",action="store_true",help="Only emit kernels; do not emit kernel launchers and do not modify host code [default: False].")
    group_fort2hip = parser.add_argument_group('Fortran-to-HIP')
    group_fort2hip.add_argument("-C","--emit-cpu-impl",dest="emit_cpu_implementation",action="store_true",help="Per detected loop kernel, also extract the CPU implementation  [default: (default) config value].")
    group_fort2hip.add_argument("-G","--emit-debug-code",dest="emit_debug_code",action="store_true",help="Generate debug code into the kernel launchers that allows to print kernel arguments, launch parameters, input/output array norms and elements, or to synchronize a kernel [default: (default) config value].")
    
    # CUDA Fortran
    group_cuf = parser.add_argument_group('CUDA Fortran input')
    group_cuf.add_argument("--cublas-v2",dest="cublasV2",action="store_true",help="Assume cublas v2 function signatures that use a handle. Overrides config value.")
    
    # developer options
    group_developer = parser.add_argument_group('Developer options (logging, profiling, ...)')
    group_developer.add_argument("-v","--verbose",dest="verbose",required=False,action="store_true",help="Print all log messages to error output stream too.")
    group_developer.add_argument("--log-level",dest="log_level",required=False,type=str,default="",help="Set log level. Overrides config value.")
    group_developer.add_argument("--log-filter",dest="log_filter",required=False,type=str,default=None,help="Filter the log output according to a regular expression.")
    group_developer.add_argument("--log-traceback",dest="log_traceback",required=False,action="store_true",help="Append gpufort traceback information to the log when encountering warning/error.")
    group_developer.add_argument("--dump-linemaps",dest="dump_linemaps",required=False,action="store_true",help="Write the lines-to-statements mappings to disk pre & post applying code transformations.")
    group_developer.add_argument("--prof",dest="profiling_enable",required=False,action="store_true",help="Profile gpufort.")
    group_developer.add_argument("--prof-num-functions",dest="profiling_num_functions",required=False,type=int,default=50,help="The number of python functions to include into the summary [default=50].")
    group_developer.add_argument("--create-gpufort-headers",dest="create_gpufort_headers",action="store_true",help="Generate the GPUFORT header files.")
    group_developer.add_argument("--create-gpufort-sources",dest="create_gpufort_sources",action="store_true",help="Generate the GPUFORT source files.")

    parser.set_defaults(print_config_defaults=False,
      dump_linemaps=False,wrap_in_ifdef=False,cublasV2=False,
      only_emit_kernels_and_launchers=False,only_emit_kernels=False,only_modify_translation_source=False,\
      emit_cpu_implementation=False,emit_debug_code=False,\
      create_gpufort_headers=False,create_gpufort_sources=False,\
      print_gfortran_config=False,print_cpp_config=False,\
      print_ldflags=False,print_ldflags_gpufort_rt=False,
      only_create_gpufort_module_files=False,skip_create_gpufort_module_files=False,verbose=False,\
      log_traceback=False,profiling_enable=False)
    args, unknown_args = parser.parse_known_args()

    ## Simple output commands
    if args.print_path:
        print(__GPUFORT_ROOT_DIR,file=sys.stdout)
        sys.exit()
    hip_platform=os.environ.get("HIP_PLATFORM","amd")
    if args.print_cpp_config:
        cpp_config  = "-D"+linemapper.LINE_GROUPING_IFDEF_MACRO
        cpp_config += " -I"+os.path.join(__GPUFORT_ROOT_DIR,"include")
        cpp_config += " -I"+os.path.join(__GPUFORT_ROOT_DIR,"include",hip_platform)
        print(cpp_config,file=sys.stdout)
        sys.exit()
    if args.print_gfortran_config:
        fortran_config  = " -cpp -std=f2008 -ffree-line-length-none"
        fortran_config += " -D"+linemapper.LINE_GROUPING_IFDEF_MACRO
        fortran_config += " -I"+os.path.join(__GPUFORT_ROOT_DIR,"include")
        fortran_config += " -I"+os.path.join(__GPUFORT_ROOT_DIR,"include",hip_platform)
        print(fortran_config,file=sys.stdout)
        sys.exit()
    if args.print_ldflags:
        ldflags = " -L"+os.path.join(__GPUFORT_ROOT_DIR,"lib") + " -lgpufort_"+hip_platform
        print(ldflags,file=sys.stdout)
        sys.exit()
    if args.print_ldflags_gpufort_rt:
        ldflags  = " -L"+os.path.join(__GPUFORT_ROOT_DIR,"lib") + " -lgpufort_acc_"+hip_platform
        ldflags += " -lgpufort_"+hip_platform
        print(ldflags,file=sys.stdout)
        sys.exit()
    call_exit = False
    if args.create_gpufort_headers:
        fort2x.hip.generate_gpufort_headers(os.getcwd())
        call_exit = True
    if args.create_gpufort_sources:
        fort2x.hip.generate_gpufort_sources(os.getcwd())
        call_exit = True
    if call_exit:
        sys.exit()
    if args.print_config_defaults:
        gpufort_python_dir=os.path.dirname(os.path.realpath(__file__))
        options_files = [
          "./gpufort_options.py.in",
          "scanner/scanner_options.py.in",
          "translator/translator_options.py.in",
          "fort2hip/fort2hip_options.py.in",
          "indexer/indexer_options.py.in",
          "indexer/scoper_options.py.in",
          "linemapper/linemapper_options.py.in",
          "utils/logging_options.py.in"
        ]
        print("\nCONFIGURABLE GPUFORT OPTIONS (DEFAULT VALUES):")
        for options_file in options_files:
            prefix = options_file.split("/")[1].split("_")[0]
            prefix = prefix.replace("logging","utils.logging") # hack
            print("\n---- "+prefix+" -----------------------------")
            with open(gpufort_python_dir+"/"+options_file) as f:
                for line in f.readlines():
                   if len(line.strip()) and\
                     "# SPDX-License-Identifier:" not in line and\
                     "# Copyright (c)" not in line:
                       if line[0] not in [" ","#","}","]"] and "=" in line:
                           parts = line.split("=")
                           if prefix == "gpufort":
                               key   = (parts[0].rstrip()).ljust(36)
                           else:
                               key   = (prefix+"."+parts[0].rstrip()).ljust(36)
                           value = parts[1].lstrip()
                           print(key+"= "+value,end="")
                       else:
                           print(line,end="")
        print("")
        sys.exit(0)
    ## VALIDATION
    if args.only_create_gpufort_module_files and args.skip_create_gpufort_module_files:
        msg = "switches '--only-create-mod-files' and '--skip-generate-mod-files' are mutually exclusive."
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    # mutually exclusive arguments
    if ( int(args.only_emit_kernels_and_launchers) +\
         int(args.only_emit_kernels) +\
         int(args.only_modify_translation_source) )  > 1:
        msg = "switches '--only-emit-kernels', '--only-emit-kernels-and-launchers', and 'only-modify-host-code' are mutually exclusive."
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    # unknown options
    if len(unknown_args):
        msg = "unknown arguments (may be used by registered actions): {}".format(" ".join(unknown_args))
        print("WARNING: "+msg,file=sys.stderr)
    # check if input is set
    if args.input is None:
        msg = "no input file"
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    if args.input[0] != "/":
        args.input = args.working_dir + "/" + args.input 
    if not os.path.exists(args.input):
        msg = "input file '{}' cannot be found".format(args.input)
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    ## OVERWRITE CONFIG VALUES
    # parse file and create index in parallel
    if args.destination_dialect != None:
        scanner.DESTINATION_DIALECT = \
          scanner.check_destination_dialect(args.destination_dialect)
    # only create modules files / skip module file generation
    if args.only_create_gpufort_module_files: 
        ONLY_CREATE_GPUFORT_MODULE_FILES = True
    if args.skip_create_gpufort_module_files: 
        SKIP_CREATE_GPUFORT_MODULE_FILES = True
    # fort2hip
    # only generate kernels / modify source 
    if args.only_emit_kernels_and_launchers:
        ONLY_EMIT_KERNELS_AND_LAUNCHERS = True
    if args.only_emit_kernels:
        ONLY_EMIT_KERNELS = True
    if args.only_modify_translation_source:
        ONLY_MODIFY_TRANSLATION_SOURCE = True
    # wrap modified lines in ifdef
    if args.wrap_in_ifdef:
        linemapper.LINE_GROUPING_WRAP_IN_IFDEF = True
    # developer: logging
    if len(args.log_level):
        LOG_LEVEL = args.log_level
    if args.verbose:
        utils.logging.VERBOSE = True
    if args.log_traceback:
        utils.logging.TRACEBACK = True
    if args.log_filter != None:
        utils.logging.LOG_FILTER = args.log_filter
    # developer: profiling:
    if args.profiling_enable:
        PROFILING_ENABLE = True
        PROFILING_OUTPUT_NUM_FUNCTIONS = args.profiling_num_functions
    # developer: other
    if args.dump_linemaps:
        DUMP_LINEMAPS = True
    # CUDA Fortran
    if args.cublasV2:
        scanner.CUBLAS_VERSION = 2
        translator.CUBLAS_VERSION = 2
    return args, unknown_args

def init_logging(input_filepath):
    global LOG_LEVEL
    global LOG_FORMAT

    input_filepath_hash = hashlib.md5(input_filepath.encode()).hexdigest()[0:8]
    logfile_basename = "log-{}.log".format(input_filepath_hash)
   
    log_format   = LOG_FORMAT.replace("%(filename)s",input_filepath)
    log_filepath = utils.logging.init_logging(logfile_basename,log_format,LOG_LEVEL)
 
    msg = "input file: {0} (log id: {1})".format(input_filepath,input_filepath_hash)
    utils.logging.log_info(LOG_PREFIX,"init_logging",msg)
    msg = "log file:   {0} (log level: {1}) ".format(log_filepath,LOG_LEVEL)
    utils.logging.log_info(LOG_PREFIX,"init_logging",msg)
    return log_filepath

if __name__ == "__main__":
    # read config and command line arguments
    config_filepath, include_dirs, defines = parse_raw_command_line_arguments()
    if config_filepath != None:
        parse_config(config_filepath)
    args, unknown_args = parse_cl_args()
    if len(POST_CLI_ACTIONS):
        msg = "run registered actions"
        utils.logging.log_info(msg,verbose=False)
        for action in POST_CLI_ACTIONS:
            if callable(action):
                action(args,unknown_args)

    # init logging
    input_filepath = os.path.abspath(args.input)
    log_filepath   = init_logging(input_filepath)
    
    # Update INCLUDE_DIRS from all sources    
    INCLUDE_DIRS += args.search_dirs
    INCLUDE_DIRS += include_dirs
    one_or_more_search_dirs_not_found = False
    for i,directory in enumerate(INCLUDE_DIRS):
        if directory[0]  != "/":
            INCLUDE_DIRS[i] = args.working_dir+"/"+directory
        if not os.path.exists(INCLUDE_DIRS[i]):
            msg = "search directory '{}' cannot be found".format(INCLUDE_DIRS[i])
            utils.logging.log_error(msg,verbose=False) 
            one_or_more_search_dirs_not_found = True
    INCLUDE_DIRS.append(args.working_dir)
    if one_or_more_search_dirs_not_found:
        sys.exit(2)

    # scanner must be invoked after index creation
    if PROFILING_ENABLE:
        profiler = cProfile.Profile()
        profiler.enable()
    #
    linemaps = linemapper.read_file(input_filepath,defines)
    
    if DUMP_LINEMAPS:
        utils.logging.log_info(LOG_PREFIX,"__main__","dump linemaps (before translation)")
        linemapper.dump_linemaps(linemaps,input_filepath+"-linemaps-pre.json")
    
    index   = create_index(INCLUDE_DIRS,defines,input_filepath,linemaps)
    if not ONLY_CREATE_GPUFORT_MODULE_FILES:
        # configure fort2hip
        if ONLY_EMIT_KERNELS_AND_LAUNCHERS:
            fort2x.hip.EMIT_KERNEL_LAUNCHER = True
        if ONLY_EMIT_KERNELS:
            fort2x.hip.EMIT_KERNEL_LAUNCHER = False
        if args.emit_cpu_implementation:
            fort2x.hip.EMIT_CPU_IMPLEMENTATION = True
        if args.emit_debug_code:
            fort2x.hip.EMIT_DEBUG_CODE = True
        stree   = scanner.parse_file(linemaps,index,input_filepath)    
 
        # extract kernels
        if "hip" in scanner.DESTINATION_DIALECT: 
            kernels_to_convert_to_hip = ["*"]
        else:
            kernels_to_convert_to_hip = scanner.KERNELS_TO_CONVERT_TO_HIP
        fortran_module_filepath, main_hip_filepath =\
          fort2x.hip.generate_hip_files(stree,index,kernels_to_convert_to_hip,input_filepath,\
           generate_code=not ONLY_MODIFY_TRANSLATION_SOURCE)
        # modify original file
        if fortran_module_filepath != None:
            preamble = "#include \"{}\"".format(\
              os.path.basename(fortran_module_filepath))
        else:
            preamble = None
        if not (ONLY_EMIT_KERNELS or ONLY_EMIT_KERNELS_AND_LAUNCHERS):
            _intrnl_translate_source(input_filepath,stree,linemaps,index,preamble) 
    #
    if PROFILING_ENABLE:
        profiler.disable() 
        s = io.StringIO()
        sortby = 'cumulative'
        stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        stats.print_stats(PROFILING_OUTPUT_NUM_FUNCTIONS)
        print(s.getvalue())

    if DUMP_LINEMAPS:
        utils.logging.log_info(LOG_PREFIX,"__main__","dump linemaps (after translation)")
        linemapper.dump_linemaps(linemaps,input_filepath+"-linemaps-post.json")

    # shutdown logging
    msg = "log file:   {0} (log level: {1}) ".format(log_filepath,LOG_LEVEL)
    utils.logging.log_info(LOG_PREFIX,"__main__",msg)
    utils.logging.shutdown()
