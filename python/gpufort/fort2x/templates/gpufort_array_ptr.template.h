{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{% import "gpufort_array_ptr.macros.h" as gapm %}
{########################################################################################}
{{ gm.render_begin_header("GPUFORT_ARRAY_PTR_H") }}
#include "gpufort_dope.h"

{{ gapm.render_gpufort_array_ptrs(max_rank) }}
{{ gm.render_end_header("GPUFORT_ARRAY_PTR_H") }}
