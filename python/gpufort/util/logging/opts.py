# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
log_format     = "%(levelname)s: %(message)s" # log format; '$(<var>)s' are placeholders
log_dir        = "/tmp/gpufort"               # directory for storing log files
log_dir_create = True                         # create log dir if it does not exist
verbose        = False                        # print to stderr too (w.r.t. to chosen log level)

log_filter     = None                         # a regular expression or string that a substring of the log output must match; set this value to None if no log filtering should be applied.

traceback  = False