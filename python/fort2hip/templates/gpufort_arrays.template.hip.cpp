{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "templates/gpufort_arrays.macros.h" as gam %}
// This file was generated from a template via gpufort --gpufort-create-headers
// C bindings
#include "gpufort_arrays.h"

{{ gam.gpufort_arrays_c_bindings(datatypes,max_rank) }}
