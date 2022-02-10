# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
__GPUFORT_INC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"../include")
include_dirs = [ __GPUFORT_INC_DIR ]
        # Include dir for gpufort_mod file and header files.

post_cli_actions = [] 
       # List of actions to run after parsing the command line arguments
       # Note that config is read before parsing other command line arguments.
       # Register functions with parameters (args,unknown_args) (see: argparse.parse_known_args

# logging
log_level  = "warning"
        # Log level; only print warnings by default.
log_format = "[%(levelname)s][infile:%(filename)s] gpufort:%(message)s"
        # Log format
log_prefix = "gpufort"
        # Prefix for all log output

only_create_gpufort_module_files = False
        # Only create and write GPUFORT module files and no other output.
skip_create_gpufort_module_files = False 
        # Skip creating and writing GPUFORT module files.

only_modify_translation_source  = False
        # Do only modify the translation source.
only_emit_kernels_and_launchers = False
        # Do only emit/extract HIP C++ kernels and respective launchers. (Only makes sense if destination language is HIP.)
only_emit_kernels               = False
        # Do only emit/extract HIP C++ kernels but no launchers. (Only makes sense if destination language is HIP.)

modified_file_ext = "-gpufort.f08"
       # Suffix for the modified file.

prettify_modified_translation_source = False 
        # Prettify the translation source after all modifications have been applied.
        # (Does not change the actual source but the modified version of it.)

profiling_enable               = False
        # Enable profiling of GPUFORT
profiling_output_num_functions = 50
        # Number of functions to output when profiling GPUFORT

dump_linemaps = False
        # Write the lines-to-statements mappings to disk before & after
        # applying code transformations; suffix="-linemaps-(pre|post).json"