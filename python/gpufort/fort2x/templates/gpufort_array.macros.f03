{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{% import "common.macros.f03" as cm %}
{########################################################################################}
{%- macro array_stride(r,array_rank) -%}
{%- if r == 1 -%}
1
{%- elif r == array_rank+1 -%}
array%data%num_elements
{%- else -%}
array%data%stride{{r}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_module_procedure(prefix,
                                  max_rank_ub,
                                  routine,
                                  datatypes) -%}
module procedure :: &{{"\n"}}
{%- for rank in range(1,max_rank_ub) -%}
{%-   set size_dims = ",dimension("+rank|string+")" -%}
{%-   set f_array  = prefix+rank|string -%}
{%-   set binding  = f_array+"_"+routine -%}
{%-   for tuple in datatypes -%}
{{binding | indent(4,True)}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}
{%-   endfor -%}{{",&\n" if not loop.last}}
{%- endfor %}{{""}}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_data_access_interfaces(datatypes,max_rank) -%}
{% set prefix      = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for routine in ["num_elements","num_data_bytes"] %}
{% set iface       = prefix+"_"+routine %}
!>
!> \return the number of {{"array elements" if routine=="num_elements" else "bytes required for storing the array elements"}}.
!>
interface {{iface}}
  module procedure :: &
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
    {{binding}}{{",&" if not loop.last}}
{% endfor %}
end interface
{% endfor %}{# routine #}

{% for loc in ["host","dev"] %}
{% set routine = (loc+"ptr") | replace("dev","device") %}
{% set iface = prefix+"_"+routine %}
!>
!> Writes {{ loc | replace("dev","device") }} address of the array data
!> and this arrays bounds to \p data_{{loc}}.
!>
interface {{iface}}
  module procedure :: &
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% for tuple in datatypes %}
    {{binding}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %}
end interface
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_data_access_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
function {{f_array}}_num_data_bytes(array) result(retval)
  implicit none
  type({{f_array}}),intent(in) :: array
  integer(c_size_t) :: retval 
  !
  retval = int({{ array_stride(rank+1,rank) }},c_size_t)*int(array%bytes_per_element,c_size_t)
end function

function {{f_array}}_num_elements(array) result(retval)
  implicit none
  type({{f_array}}),intent(in) :: array
  integer(c_int) :: retval 
  !
  retval = {{ array_stride(rank+1,rank) }}
end function
{% endfor %}

{% for loc in ["host","dev"] %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array = prefix+rank|string %}
{% set routine = (loc+"ptr") | replace("dev","device") %}
{% set binding = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% for tuple in datatypes %}
subroutine {{binding}}_{{tuple.f_kind}}(&
    array,&
    data_{{loc}})
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(in) :: array
  {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_{{loc}} 
  !
  integer(c_int) :: offset_remainder
  integer(c_int) :: {{ gm.separated_list_single_line("n",",",rank) }},&
                    {{ gm.separated_list_single_line("lb",",",rank) }}
  !
  {{tuple.f_type}},pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: tmp
  !
  {{ gm.separated_list_single_line("n"," = 1; ",rank) }} = 1
  {{ gm.separated_list_single_line("lb"," = 1; ",rank) }} = 1
  ! 
{% for i in range(rank,0,-1) %}
  n{{i}} = {{ array_stride(i+1,rank) }} / {{ array_stride(i,rank) }}
{% endfor %}
  !
  call c_f_pointer(array%data%data_{{loc}},tmp,SHAPE=[{{ gm.separated_list_single_line("n",",",rank) }}])
  !
  offset_remainder = array%data%index_offset
{% for i in range(rank,0,-1) %}
  lb{{i}} = -offset_remainder / {{ array_stride(i,rank) }}
  offset_remainder = offset_remainder + lb{{i}}*{{ array_stride(i,rank) }}
{% endfor %}
  !
  data_{{loc}}({{ gm.separated_list_single_line("lb",":,",rank) }}:) => tmp
end subroutine
{% endfor %}
{% endfor %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_init_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set routine = "init" %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
function {{binding}}{{async_suffix}}_host(&
    array,&
    bytes_per_element,&
    sizes,lbounds,&
    {{"stream," if is_async}}alloc_mode,sync_mode) result(ierr)
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(inout) :: array
  integer(c_int){{size_dims}},intent(in) :: sizes
  integer(c_int){{size_dims}},intent(in),optional :: lbounds{{"
  type(c_ptr),intent(in) :: stream" if is_async}}
  integer(c_int),intent(in) :: bytes_per_element
  integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),optional :: alloc_mode 
  integer(kind(gpufort_array_sync_none)),intent(in),optional :: sync_mode 
  !
  integer(kind(hipSuccess)) :: ierr
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  integer(kind(gpufort_array_wrap_host_wrap_device)) :: opt_alloc_mode 
  integer(kind(gpufort_array_sync_none)) :: opt_sync_mode 
  !
  opt_lbounds    = 1
  opt_alloc_mode = gpufort_array_alloc_host_alloc_device ! allocate both by default
  opt_sync_mode  = gpufort_array_sync_none
  if ( present(lbounds) )    opt_lbounds    = lbounds             
  if ( present(alloc_mode) ) opt_alloc_mode = alloc_mode 
  if ( present(sync_mode) )  opt_sync_mode  = sync_mode 
  ierr = {{binding}}{{async_suffix}}(&
    array,&
    bytes_per_element,&
    c_null_ptr,c_null_ptr, &
    sizes, opt_lbounds,&
    {{"stream," if is_async}}opt_alloc_mode, opt_sync_mode )
end function
{% for tuple in datatypes %}
function {{binding}}{{async_suffix}}_{{tuple.f_kind}}(&
    array,&
    data_host,lbounds,&
    {{"stream," if is_async}}data_dev,alloc_mode,sync_mode) result(ierr)
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(inout) :: array
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_host 
  integer(c_int){{size_dims}},intent(in),optional :: lbounds{{"
  type(c_ptr),intent(in) :: stream" if is_async}}
  type(c_ptr),intent(in),optional :: data_dev
  integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),optional :: alloc_mode 
  integer(kind(gpufort_array_sync_none)),intent(in),optional :: sync_mode 
  !
  integer(kind(hipSuccess)) :: ierr
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  type(c_ptr) :: opt_data_dev
  integer(kind(gpufort_array_wrap_host_wrap_device)) :: opt_alloc_mode 
  integer(kind(gpufort_array_sync_none)) :: opt_sync_mode 
  !
  opt_lbounds    = 1
  opt_data_dev   = c_null_ptr
  opt_alloc_mode = gpufort_array_wrap_host_alloc_device
  opt_sync_mode  = gpufort_array_sync_none
  if ( present(lbounds) )    opt_lbounds    = lbounds             
  if ( present(data_dev) )   opt_data_dev   = data_dev              
  if ( present(alloc_mode) ) opt_alloc_mode = alloc_mode 
  if ( present(sync_mode) )  opt_sync_mode  = sync_mode 
  ierr = {{binding}}{{async_suffix}}(&
    array,&
    int({{tuple.bytes}}, c_int),&
    c_loc(data_host),opt_data_dev,&
    shape(data_host), opt_lbounds,&
    {{"stream," if is_async}}opt_alloc_mode, opt_sync_mode)
end function
{% endfor %}{# datatypes #}
{% endfor %}{# async_suffix #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_copy_to_from_buffer_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{%   set f_array  = prefix+rank|string %}
{%   for routine in ["copy_from_buffer","copy_to_buffer"] %}
{%     set binding  = f_array+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
{%       for async_suffix in ["_async",""] %}
{%         set is_async = async_suffix == "_async" %}
  function {{binding}}{{async_suffix}}_{{tuple.f_kind}}(&
    array,buffer,memcpy_kind{{",&
    stream" if is_async}}) &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(inout) :: array
    {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
    integer(kind(hipMemcpyHostToDevice)),intent(in),value :: memcpy_kind{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    integer(kind(hipSuccess)) :: ierr
    !
    ierr = {{binding}}{{async_suffix}}(&
      array,c_loc(buffer),memcpy_kind{{",stream" if is_async}})
  end function
{%       endfor %}{# async_suffix #}
{%     endfor %}{# datatypes #}
{%   endfor %}{# routines #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_allocate_buffer_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set routine = "allocate_"+target+"_buffer" %}
{%     set binding  = f_array+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
  function {{binding}}_{{tuple.f_kind}}(&
      array,buffer{{",&
      pinned,flags" if is_host}}) &
        result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(in) :: array
    {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
    logical(c_bool),value,optional,intent(in) :: pinned
    integer(c_int),value,optional,intent(in) :: flags
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
{%     if is_host %}
    !
    logical(c_bool) :: opt_pinned
    integer(c_int) :: opt_flags
{%     endif %}
{%     if is_host %}
    !
    opt_pinned = .true.
    opt_flags  = 0
    if ( present(pinned) ) opt_pinned = pinned
    if ( present(flags) ) opt_flags = flags
{%     endif      %}
    !
    ierr = {{binding}}(&
      array,c_loc(buffer){{",opt_pinned,opt_flags" if is_host}})
{% set rank_ub=rank+1 %}
    buffer(&
{% for i in range(1,rank_ub) %}
      gpufort_array{{rank}}_lbound(array,{{i}}):{{ "," if not loop.last else ")" }}&
{% endfor %}
        => buffer
  end function
{%     endfor %}{# datatypes #}
{%   endfor %}{# rank #}
{% endfor %}{# target #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_deallocate_buffer_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set routine = "deallocate_"+target+"_buffer" %}
{%     set binding  = f_array+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
  function {{binding}}_{{tuple.f_kind}}(&
      array,buffer{{",&
      pinned" if is_host}}) &
        result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(in) :: array
    {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
    logical(c_bool),value,optional,intent(in) :: pinned
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
{%     if is_host %}
    !
    logical(c_bool) :: opt_pinned
{%     endif %}
{%     if is_host %}
    !
    opt_pinned = .true.
    if ( present(pinned) ) opt_pinned = pinned
{%     endif      %}
    !
    ierr = {{binding}}(&
      array,c_loc(buffer){{",pinned" if is_host}})
  end function
{%     endfor %}{# datatypes #}
{%   endfor %}{# rank #}
{% endfor %}{# target #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_wrap_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{%   set f_array  = prefix+rank|string %}
{%   set routine = "init" %}
{%   set binding  = f_array+"_"+routine %}
{%   set size_dims = ",dimension("+rank|string+")" %}
{%   for tuple in datatypes %}
{%     for dev in ["","_cptr"] %} 
{%       for async_suffix in ["_async",""] %}
{%         set is_async = async_suffix == "_async" %}
function {{f_array}}_wrap{{async_suffix}}_{{tuple.f_kind}}{{dev}}(&
    data_host,data_dev,lbounds,&
    {{"stream," if is_async}}sync_mode,ierr) result(array)
  use iso_c_binding
  use hipfort_enums
  implicit none
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_host 
{%         if dev == "_cptr" %}
  type(c_ptr),intent(in) :: data_dev
{%         else %}
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_dev 
{%         endif %}
  integer(c_int){{size_dims}},intent(in),optional :: lbounds{{" 
  type(c_ptr),intent(in) :: stream" if is_async}}
  integer(kind(gpufort_array_sync_none)),intent(in),optional :: sync_mode
  integer(kind(hipSuccess)),intent(inout),optional :: ierr
  !
  type({{f_array}}) :: array
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  integer(kind(gpufort_array_sync_none)) :: opt_sync_mode 
  integer(kind(hipSuccess)) :: opt_ierr
  !
  opt_lbounds    = 1
  opt_sync_mode  = gpufort_array_sync_none
  if ( present(lbounds) )   opt_lbounds    = lbounds             
  if ( present(sync_mode) ) opt_sync_mode  = sync_mode 
{%         if dev == "_cptr" %}
{%           set data_dev_arg = "data_dev" %}
{%         else %}
{%           set data_dev_arg = "c_loc(data_dev)" %}
{%         endif %}
  opt_ierr = {{binding}}{{async_suffix}}(&
    array,&
    int({{tuple.bytes}}, c_int),&
    c_loc(data_host),{{data_dev_arg}},&
    shape(data_host), opt_lbounds {{",stream" if is_async}},&
    gpufort_array_wrap_host_wrap_device,&
    opt_sync_mode)
  if ( present(ierr) ) ierr = opt_ierr
end function
{%         endfor %}{# asyncr #}
{%       endfor %}{# dev #}
{%     endfor %}{# datatypes #}
{%     set routine = "init" %}
{%     set binding  = f_array+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
function {{f_array}}_wrap_device_cptr(&
    data_dev,sizes,lbounds,bytes_per_element) result(array)
  use iso_c_binding
  use hipfort_enums
  implicit none
  type(c_ptr),intent(in) :: data_dev 
  integer(c_int){{size_dims}},intent(in) :: sizes
  integer(c_int){{size_dims}},intent(in),optional :: lbounds
  integer(c_int),intent(in),optional :: bytes_per_element
  !
  type({{f_array}}) :: array
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  integer(c_int) :: opt_bytes_per_element
  !
  integer(kind(hipSuccess)) :: ierr
  !
  opt_lbounds = 1
  opt_bytes_per_element = 1
  if ( present(lbounds) ) opt_lbounds = lbounds             
  if ( present(bytes_per_element) ) opt_bytes_per_element = bytes_per_element             
  ierr = {{binding}}(&
    array,&
    opt_bytes_per_element,&
    c_null_ptr,data_dev,&
    sizes, opt_lbounds,&
    gpufort_array_wrap_host_wrap_device,&
    gpufort_array_sync_none)
end function
{%     for tuple in datatypes %}
function {{f_array}}_wrap_device_ptr_{{tuple.f_kind}}(&
    data_dev,lbounds) result(array)
  use iso_c_binding
  use hipfort_enums
  implicit none
  {{tuple.f_type}},intent(in),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_dev 
  integer(c_int){{size_dims}},intent(in),optional :: lbounds
  !
  type({{f_array}}) :: array
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  !
  integer(kind(hipSuccess)) :: ierr
  !
  opt_lbounds    = 1
  if ( present(lbounds) ) opt_lbounds = lbounds             
  ierr = {{binding}}(&
    array,&
    int({{tuple.bytes}}, c_int),&
    c_null_ptr,c_loc(data_dev),&
    shape(data_dev), opt_lbounds,&
    gpufort_array_wrap_host_wrap_device,&
    gpufort_array_sync_none)
end function
{%   endfor %}{# datatypes #}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_interfaces(datatypes,max_rank) -%}
{% set max_rank_ub = max_rank+1 %}
{% set prefix = "gpufort_array" %}
{# INIT #}
{% set routine = "init" %}
{% set iface = prefix+"_"+routine %}
!> Initialize a gpufort_array of a rank that matches that of the input data.
{% for async_suffix in ["_async",""] %}
{%   set is_async = async_suffix == "_async" %}
interface {{iface}}{{async_suffix}}
{%   for rank in range(1,max_rank_ub) %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}} (&
      array,&
      bytes_per_element,&
      data_host,data_dev,&
      sizes, lbounds,&
      {{"stream," if is_async}}alloc_mode,sync_mode) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    import gpufort_array_wrap_host_wrap_device
    import gpufort_array_sync_none
    implicit none
    type({{f_array}}),intent(inout) :: array
    integer(c_int),intent(in),value :: bytes_per_element
    type(c_ptr),intent(in),value :: data_host, data_dev
    integer(c_int){{size_dims}},intent(in) :: sizes, lbounds
    integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),value :: alloc_mode 
    integer(kind(gpufort_array_sync_none)),intent(in),value :: sync_mode{{"
    type(c_ptr),intent(in),value :: stream" if is_async}}
    !
    integer(kind(hipSuccess)) :: ierr
  end function
{%   endfor %}
    module procedure :: &
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+routine %}
      {{binding}}{{async_suffix}}_host,&
{%     for tuple in datatypes %}
      {{binding}}{{async_suffix}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{%   endfor %}
end interface
{% endfor %}{# async_suffix #}
{# WRAP #}
{% for async_suffix in ["_async",""] %}
{%   set is_async = async_suffix == "_async" %}
{%   set routine = "wrap" %}
{%   set iface = prefix+"_"+routine %}
!> Wrap a Fortran or type(c_ptr) pointer pair that points to host and device data.
!> Only wrap, allocate nothing,
!> and do not synchronize at initialization or destruction time.
!> \return gpufort_array of a rank that matches that of the input data.
interface {{iface}}{{async_suffix}}
    module procedure :: &{{"\n"}}
{%-  for rank in range(1,max_rank_ub) -%}
{%-    set size_dims = ",dimension("+rank|string+")" -%}
{%-    set f_array  = prefix+rank|string -%}
{%-    set binding  = f_array+"_"+routine -%}
{%-    for tuple in datatypes -%}
{%-      for dev in ["","_cptr"] -%} 
{{binding | indent(6,True)}}{{async_suffix}}_{{tuple.f_kind}}{{dev}}{{",&\n" if not loop.last}}
{%-      endfor -%}{{",&\n" if not loop.last}}
{%-    endfor -%}{{",&\n" if not loop.last}}
{%-  endfor %}{{""}}
end interface
{% endfor %}{# async_suffix #}

{% set routine = "wrap_device_ptr" %}
{% set iface = prefix+"_"+routine %}
!> Wrap a Fortran pointer or type(c_ptr) that points to device data.
!> Set the host pointer to null. Only wrap, allocate nothing,
!> and do not synchronize at initialization or destruction time.
!> \note Calling gpufort_array_copy_to_host or gpufort_array_copy_to_device
!>       will result in HIP errors as the host pointer is set to null.
!> \return gpufort_array of a rank that matches that of the input data.
interface {{iface}}
    module procedure :: &
{% for rank in range(1,max_rank_ub) %}
{%   set size_dims = ",dimension("+rank|string+")" %}
{%   set f_array  = prefix+rank|string %}
{%   set binding  = f_array+"_"+routine %}
{%   for tuple in datatypes %}
      {{binding}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %}
end interface

{% for async_suffix in ["_async",""] %}
{%   set is_async = async_suffix == "_async" %}
{# DESTROY, COPY_TO_HOST, COPY_TO_DEVICE #}
{%   for routine in ["destroy","copy_to_host","copy_to_device"] %}
{%     set iface = prefix+"_"+routine %}
interface {{iface}}{{async_suffix}}
{%     for rank in range(1,max_rank_ub) %}
{%       set f_array  = prefix+rank|string %}
{%       set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}}(array{{",stream" if is_async}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: array{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    integer(kind(hipSuccess)) :: ierr
  end function
{%     endfor %}
end interface
{%   endfor %}
{# COPY_FROM_BUFFER, COPY_TO_BUFFER #}
{%   for routine in ["copy_from_buffer","copy_to_buffer"] %}
{%     set iface = prefix+"_"+routine %}
!>
{%     if routine == "copy_to_buffer" %}
!> Copy this array's host or device data TO a host or device buffer.
{%     else %}
!> Copy this array's host or device data FROM a host or device buffer.
{%     endif %}
!>
interface {{iface}}{{async_suffix}}
{%     for rank in range(1,max_rank_ub) %}
{%       set f_array  = prefix+rank|string %}
{%       set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}}(&
    array,buffer,memcpy_kind{{",&
    stream" if is_async}}) &
        bind(c,name="{{binding}}{{async_suffix}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: array
    type(c_ptr),value,intent(in) :: buffer
    integer(kind(hipMemcpyHostToDevice)),intent(in),value :: memcpy_kind{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    integer(kind(hipSuccess)) :: ierr
  end function
{%     endfor %}
{{ render_module_procedure(prefix,
                           max_rank_ub,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{%   endfor %}
{# DEC_NUM_REFS #}
{%   set routine = "dec_num_refs" %}
{%   set iface = prefix+"_"+routine %}
interface {{iface}}{{async_suffix}}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}}(&
    array,destroy_if_zero_refs{{",&
    stream" if is_async}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: array{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    logical(c_bool),intent(in),value :: destroy_if_zero_refs
    integer(kind(hipSuccess)) :: ierr
  end function
{%   endfor %}
end interface
{% endfor %}{# async_suffix #}
{# ALLOCATE_BUFFER #}
{% for target in ["host","device"] %}
{%   set is_host = target == "host" %}
{%   set routine = "allocate_"+target+"_buffer" %}
{%   set iface = prefix+"_"+routine %}
!> Allocate a {{target}} buffer with
!> the same size as the data buffers
!> associated with this gpufort array.
!> @see num_data_bytes()
!> \param[inout] pointer to the buffer to allocate
{%   if is_host %}
!> \param[in] pinned If the memory should be pinned.
!> \param[in] flags  Flags to pass to host memory allocation.
{%   endif %}
interface {{iface}}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+routine %}
  function {{binding}}(&
      array,buffer{{",pinned,flags" if is_host}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(in) :: array
    type(c_ptr),intent(in) :: buffer
{%     if is_host %}
    logical(c_bool),value :: pinned
    integer(c_int),value :: flags
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
  end function
{%   endfor %}
{{ render_module_procedure(prefix,
                           max_rank_ub,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{% endfor %}
{# DEALLOCATE_BUFFER #}
{% for target in ["host","device"] %}
{%   set is_host = target == "host" %}
{%   set routine = "deallocate_"+target+"_buffer" %}
{%   set iface = prefix+"_"+routine %}
!> Deallocate a {{target}} buffer
!> created via the allocate_{{target}}_buffer routine.
!> \see num_data_bytes(), allocate_{{target}}_buffer
!> \param[inout] the buffer to deallocate
{%   if is_host %}
!> \param[in] pinned If the memory to deallocate is pinned.
{%   endif %}
interface {{iface}}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+routine %}
  function {{binding}}(&
      array,buffer{{",pinned" if is_host}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(in) :: array
    type(c_ptr),value,intent(in) :: buffer
{%     if is_host %}
    logical(c_bool),value :: pinned
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
  end function
{%   endfor %}
{{ render_module_procedure(prefix,
                           max_rank_ub,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{% endfor %}
{# INC_NUM_REFS #}
{% set routine = "inc_num_refs" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{%   set f_array  = prefix+rank|string %}
{%   set binding  = f_array+"_"+routine %}
  function {{binding}}(array) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: array
    integer(kind(hipSuccess)) :: ierr
  end function
{% endfor %}
end interface
{# SIZE,LBOUND,UBOUND #}
{% for routine in ["size","lbound","ubound"] %}
{%   set iface = prefix+"_"+routine %}
interface {{iface}}
{%   for rank in range(1,max_rank_ub) %}
{%     set f_array = prefix+rank|string %}
{%     set binding = f_array+"_"+routine %}
  function {{binding}}(array,d) &
        bind(c,name="{{binding}}") &
          result(retval)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(in) :: array
    integer(c_int),intent(in),value :: d
    !
    integer(c_int) :: retval
  end function
{%   endfor %}
end interface
{% endfor %}
{# COLLAPSE #}
!> 
!> Collapse the array by fixing indices.
!> \return A gpufort array of reduced rank.
!> \param[in] i2,i3,... indices to fix.
!> 
{% set routine = "collapse" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set rank_ub = rank+1 %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
{% for d in range(rank-1,0,-1) %}
{% set f_array_collapsed  = prefix+d|string %}
  subroutine {{binding}}_{{d}} (collapsed_array,array,&
{{"" | indent(6,True)}}{% for e in range(d+1,rank_ub) %}i{{e}}{{"," if not loop.last}}{%- endfor %}) &
      bind(c,name="{{binding}}_{{d}}")
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    import {{f_array_collapsed}}
    implicit none
    type({{f_array_collapsed}}),intent(inout) :: collapsed_array
    type({{f_array}}),intent(in) :: array
    integer(c_int),value,intent(in) :: &
{{"" | indent(6,True)}}{% for e in range(d+1,rank_ub) %}i{{e}}{{"," if not loop.last else "\n"}}{%- endfor %}
  end subroutine
{% endfor %}{# d #}
{% endfor %}{# rank #}
end interface
{%- endmacro -%}
{########################################################################################}