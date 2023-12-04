#SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys

from gpufort import util

from .. import opts

supported_destination_dialects = set()
postprocess_backends = []

def register_postprocess_backend(src_dialect, dest_dialects, func):
    postprocess_backends.append((src_dialect, dest_dialects, func))

def check_destination_dialect(destination_dialect):
    if destination_dialect in supported_destination_dialects:
        return destination_dialect
    else:
        msg = "destination dialect '{}' is not supported. Must be one of: {}".format(
            destination_dialect, ", ".join(supported_destination_dialects))
        raise ValueError(msg)

@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess(stree, index, **kwargs):
    """Add use statements as well as handles plus their creation and 
    destruction for certain math libraries.
    """
    for src_dialect, dest_dialects, func in postprocess_backends:
        if (src_dialect in opts.source_dialects and
           opts.destination_dialect in dest_dialects):
            func(stree, index)

__all__ = [
    "register_postprocess_backend",
    "postprocess",
    "supported_destination_dialects",
    "check_destination_dialect",
]
