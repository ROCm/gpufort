{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "src/gpufortrt_api.macros.f90" as gam %}
{########################################################################################}
! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt_api
  use gpufortrt_api_core
{{ gam.render_map_interfaces(datatypes,max_rank) | indent(2,True) }}

{{ gam.render_copy_interfaces(datatypes,max_rank) | indent(2,True) }}
{{ gam.render_interface("gpufortrt_delete",datatypes,max_rank) | indent(2,True) }}
{{ gam.render_interface("gpufortrt_copyout",datatypes,max_rank) | indent(2,True) }}

{{ gam.render_interface("gpufortrt_update_self",datatypes,max_rank) | indent(2,True) }}
{{ gam.render_interface("gpufortrt_update_device",datatypes,max_rank) | indent(2,True) }}

{{ gam.render_use_device_interface(datatypes,max_rank) | indent(2,True) }}

{{ gam.render_deviceptr_interfaces(datatypes,max_rank) | indent(2,True) }}

contains

{{ gam.render_map_routines(datatypes,max_rank) | indent(2,True) }}

{{ gam.render_basic_copy_routines() | indent(2,True) }}
{{ gam.render_specialized_copy_routines(datatypes,max_rank) | indent(2,True) }}
{{ gam.render_specialized_present_routines(datatypes,max_rank) | indent(2,True) }}
{{ gam.render_basic_delete_copyout_routines() | indent(2,True) }}
{{ gam.render_specialized_delete_copyout_routines(datatypes,max_rank) | indent(2,True) }}

{{ gam.render_basic_update_routine("self") }}
{{ gam.render_specialized_update_routines("self",datatypes,max_rank) }}

{{ gam.render_basic_update_routine("device") | indent(2,True) }}
{{ gam.render_specialized_update_routines("device",datatypes,max_rank) | indent(2,True) }}

{{ gam.render_specialized_use_device_routines(datatypes,max_rank) | indent(2,True) }}

{{ gam.render_specialized_deviceptr_routines(datatypes,max_rank) | indent(2,True) }}
end module
