{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort_array_ptr.macros.h" as gapm %}
{########################################################################################}
#include "gpufort_array_ptr.h"

{#
extern "C" {
  {{ gapm.render_gpufort_dope_c_bindings(max_rank) }}
}
#}
