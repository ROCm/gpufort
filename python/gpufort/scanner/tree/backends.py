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
    # TODO first part still necessary?
    #if "hip" in opts.destination_dialect or len(
    #        opts.kernels_to_convert_to_hip):
    #    # insert use statements at appropriate point
    #    def is_accelerated(child):
    #        return isinstance(child,tree.STLoopNest) or\
    #               (type(child) is tree.STProcedure and child.is_kernel_subroutine())

    #    print(stree)
    #    for stmodule in stree.find_all(filter=lambda child: isinstance(
    #            child, (tree.STModule, tree.STProgram)),
    #                                   recursively=False):
    #        module_name = stmodule.name
    #        kernels = stmodule.find_all(filter=is_accelerated,
    #                                    recursively=True)
    #        for kernel in kernels:
    #            if "hip" in opts.destination_dialect or\
    #              kernel.min_lineno() in kernels_to_convert_to_hip or\
    #              kernel.kernel_name() in kernels_to_convert_to_hip:
    #                stnode = kernel.parent.find_first(
    #                    filter=lambda child: isinstance(
    #                        child, (tree.STUseStatement, tree.STDeclaration,
    #                                tree.STContains, tree.STPlaceHolder)))
    #                assert not stnode is None
    #                indent = stnode.first_line_indent()
    #                stnode.add_to_prolog("{}use {}{}\n".format(
    #                    indent, module_name, hip_module_suffix))

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
