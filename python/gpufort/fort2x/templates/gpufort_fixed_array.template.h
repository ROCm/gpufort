{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{% import "gpufort_fixed_array.macros.h" as gfam %}
{########################################################################################}
{{ gm.render_begin_header("GPUFORT_ARRAY_PTR_H") }}
#include "gpufort_dope.h"
{{ gfam.render_gpufort_fixed_arrays(max_rank) }}
{{ gm.render_end_header("GPUFORT_ARRAY_PTR_H") }}
