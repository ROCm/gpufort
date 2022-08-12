{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "src/gpufortrt_api.macros.f90" as gam %}
{########################################################################################}
! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt
  use gpufortrt_api_core
{{ gam.render_map_interfaces(datatypes) | indent(2,True) }}

{{ gam.render_map_and_lookup_interfaces(datatypes) | indent(2,True) }}

  !> Update Directive
{{ gam.render_interface("gpufortrt_update_self",datatypes) | indent(2,True) }}
{{ gam.render_interface("gpufortrt_update_device",datatypes) | indent(2,True) }}

{{ gam.render_use_device_interface(datatypes,max_rank) }}

contains

{{ gam.render_map_routines(datatypes) | indent(2,True) }}

{{ gam.render_basic_map_and_lookup_routines() | indent(2,True) }}
{{ gam.render_specialized_map_and_lookup_routines(datatypes) | indent(2,True) }}

{{ gam.render_specialized_use_device_routines(datatypes,max_rank) | indent(2,True) }}
end module
