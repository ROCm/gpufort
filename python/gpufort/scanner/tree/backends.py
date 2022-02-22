#SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from .. import opts

SUPPORTED_DESTINATION_DIALECTS = []
POSTPROCESS_BACKENDS = []


def register_postprocess_backend(src_dialect, dest_dialect, func):
    POSTPROCESS_BACKENDS.append((src_dialect, dest_dialect, func))


def check_destination_dialect(destination_dialect):
    if destination_dialect in SUPPORTED_DESTINATION_DIALECTS:
        return destination_dialect
    else:
        msg = "scanner: destination dialect '{}' is not supported. Must be one of: {}".format(
            destination_dialect, ", ".join(SUPPORTED_DESTINATION_DIALECTS))
        util.logging.log_error(opts.log_prefix, "check_destination_dialect",
                               msg)
        sys.exit(SCANNER_ERROR_CODE)


@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess(stree, index, **kwargs):
    """Add use statements as well as handles plus their creation and 
    destruction for certain math libraries.
    """
    for src_dialect, dest_dialect, func in POSTPROCESS_BACKENDS:
        if (src_dialect in opts.source_dialects and
                dest_dialect in opts.destination_dialect):
            func(stree, index, dest_dialect)


__all__ = [
    "register_postprocess_backend",
    "postprocess",
    "SUPPORTED_DESTINATION_DIALECTS",
    "check_destination_dialect",
]
