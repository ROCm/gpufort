#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import re
import subprocess

from . import gpufortconv
from . import opts

def parse_raw_cl_args():
    """Collect host and target compiler options.
    """
    options = sys.argv[1:]
    print(options)
    
    prefix="-gpufort"
    targets=["host","target"]
    flags=["compiler","cflags","ldflags"]
    
    compiler_options = ["-".join([prefix,x,y]) for x in targets for y in flags]
    result = {}
    for copt in compiler_options:
        result[copt] = []  
    current_key     = None
    for i, opt in enumerate(list(options)):
        if opt in compiler_options:
            current_key = opt
        else:
             if current_key != None:
                 result[current_key].append(opt)
    return result

def populate_cl_arg_parser(parser):
    parser.add_argument(
        "-c",
        dest="only_compile",
        action="store_true",
        help="Only compile; do not link.",
    )
    parser.add_argument(
        "-o",
        dest="output",
        default=None,
        type=str,
        help="Path/name for the result.",
    )
    parser.add_argument(
        "-save-temps",
        dest="save_temps",
        default=os.getcwd(),
        type=str,
        help="Save temporary output files, i.e. intermediate results produced by the translation of the original code to standard Fortran plus HIP C++.",
    )
    parser.add_argument(
        "-gpufort-host-compiler",
        dest="host_compiler",
        default="gfortran",
        type=str,
        help="Compiler for host code, defaults to 'gfortran'. Use '--gpufort-host-cflags <flags>' to set compilation flags and '--gpufort-host-ldflags <flags>' to set linker flags. Flags to CPUFORT related libraries are added by default if not specified otherwise. Host and target compiler and their respective options must be specified after all other GPUFORT options.",
    )
    parser.add_argument(
        "-gpufort-target-compiler",
        dest="target_compiler",
        default="hipcc",
        type=str,
        help="Compiler for offload target code, defaults to 'hipcc'. Use '--gpufort-target-cflags <flags>' to set compilation flags and '--gpufort-target-ldflags <flags>' to set linker flags. Flags to GPUFORT related libraries are added by default if not specified otherwise. Host and target compiler and their respective options must be specified after all other GPUFORT options.",
    )
    parser.set_defaults(only_compile=False)

if __name__ == "__main__":
    config_file_path, include_dirs, defines = gpufortconv.parse_raw_cl_args()
    host_and_target_compiler_options = parse_raw_cl_args()
    if config_file_path != None:
        parse_config(config_file_path)
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="(EXPERIMENTAL) Wrapper compiler for CUDA Fortran and Fortran+OpenACC programs that relies on HIPCC.")
    populate_cl_arg_parser(parser)
    gpufortconv.populate_cl_arg_parser(parser,False)
    args, unknown_args = gpufortconv.parse_cl_args(parser,False)
    gpufortconv.map_args_to_opts(args)
    if len(opts.post_cli_actions):
        msg = "run registered actions"
        util.logging.log_info(msg, verbose=False)
        for action in opts.post_cli_actions:
            if callable(action):
                action(args, unknown_args)
    gpufortconv.set_include_dirs(args.working_dir, args.search_dirs + include_dirs)

    infile_path = os.path.abspath(args.input)
    log_file_path = gpufortconv.init_logging(os.path.abspath(args.input))
    
    fortran_file_paths, cpp_file_paths = gpufortconv.run(defines)
   
    outfile_path = args.output
    if args.only_compile and outfile_path == None:
        outfile_path = re.sub(r"\.f[0-9]*$",".o",infile_path,re.IGNORECASE))
   
    # @TODO
    subprocess.run([sys.executable, 'python1.py', ...]) 

    gpufortconv.shutdown_logging(log_file_path)
else:
    assert False, "This module is supposed to be used as standalone application!"
