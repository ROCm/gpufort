        # SPDX-License-Identifier: MIT
        # Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
log_prefix = "fort2x.gpufort_sources"
        # Prefix for log output that this component writes.
max_rank = 7
        # Generate rank-dependent classes and expressions up to
        # this dimension.
datatypes = [
    {
        "c_type": "bool",
        "f_kind": "c_bool",
        "bytes": "c_bool",
        "f_type": "logical(c_bool)"
    },
    {
        "c_type": "short",
        "f_kind": "c_short",
        "bytes": "c_short",
        "f_type": "integer(c_short)"
    },
    {
        "c_type": "int",
        "f_kind": "c_int",
        "bytes": "c_int",
        "f_type": "integer(c_int)"
    },
    {
        "c_type": "long",
        "f_kind": "c_long",
        "bytes": "c_long",
        "f_type": "integer(c_long)"
    },
    {
        "c_type": "float",
        "f_kind": "c_float",
        "bytes": "c_float",
        "f_type": "real(c_float)"
    },
    {
        "c_type": "double",
        "f_kind": "c_double",
        "bytes": "c_double",
        "f_type": "real(c_double)"
    },
    {
        "c_type": "hipFloatComplex",
        "f_kind": "c_float_complex",
        "bytes": "2*c_float_complex",
        "f_type": "complex(c_float_complex)"
    },
    {
        "c_type": "hipDoubleComplex",
        "f_kind": "c_double_complex",
        "bytes": "2*c_double_complex",
        "f_type": "complex(c_double_complex)"
    },
]
        # Generate rank-dependent classes and expressions for these
        # datatypes.
