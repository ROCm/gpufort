{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "gpufort_array_types.macros.cpp" as gatmc %}
// This file was generated from a template via gpufort --gpufort-create-headers
// C bindings
#include "gpufort_array.h"

extern "C" {
{{ gatmc.render_gpufort_array_c_bindings(thistype,max_rank) | indent(2,True) }}
{{ gatmc.render_array_property_inquiry_c_bindings(thistype,max_rank,parameterized=True) | indent(2,True) }}
}
