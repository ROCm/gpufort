{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort_dope.macros.h" as gdm %}
{########################################################################################}
#include "gpufort_dope.h"

extern "C" {
{{ gdm.render_gpufort_dope_c_bindings(max_rank) | indent(2,True) -}}
}
