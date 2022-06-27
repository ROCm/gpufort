{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{% import "gpufort_dope.macros.h" as gdm %}
{########################################################################################}
{{ gm.render_begin_header("GPUFORT_DOPE_H") }}
#include <assert.h>
#include "gpufort_array_globals.h"
{{ gdm.render_gpufort_dopes(max_rank) }}
{{ gm.render_end_header("GPUFORT_DOPE_H") }}
