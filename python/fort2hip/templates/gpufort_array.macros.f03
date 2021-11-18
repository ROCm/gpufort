{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{% import "templates/gpufort.macros.h" as gm %}
{# #}
{# #}
{# #}
{%- macro gpufort_array_fortran_data_access_interfaces(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
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
{# #}
{# #}
{# #}
{%- macro gpufort_array_fortran_data_access_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for loc in ["host","dev"] %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set routine = (loc+"ptr") | replace("dev","device") %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% for tuple in datatypes %}
subroutine {{binding}}_{{tuple.f_kind}}(&
    mapped_array,&
    data_{{loc}})
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(in)        :: mapped_array
  {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_{{loc}} 
  !
  integer(c_int)    :: offset_remainder
  integer(c_int)    :: {{ gm.separated_list_single_line("n",",",rank) }},&
                       {{ gm.separated_list_single_line("lb",",",rank) }}
  !
  {{tuple.f_type}},pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: tmp
  !
  {{ gm.separated_list_single_line("n"," = 1; ",rank) }} = 1
  {{ gm.separated_list_single_line("lb"," = 1; ",rank) }} = 1
  ! 
  n{{rank}} = int(mapped_array%data%num_elements / int(mapped_array%data%stride{{rank}},&
               c_size_t),c_int)
{% for i in range(rank-1,0,-1) %}
  n{{i}} = mapped_array%data%stride{{i+1}} / mapped_array%data%stride{{ i }}
{% endfor %}
  !
  call c_f_pointer(mapped_array%data%data_{{loc}},tmp,SHAPE=[{{ gm.separated_list_single_line("n",",",rank) }}])
  !
  offset_remainder = mapped_array%data%index_offset
{% for i in range(rank,0,-1) %}
  lb{{i}} = -offset_remainder / mapped_array%data%stride{{i}}
  offset_remainder = offset_remainder + lb{{i}}*mapped_array%data%stride{{i}}
{% endfor %}
  !
  data_{{loc}}({{ gm.separated_list_single_line("lb",":,",rank) }}:) => tmp
end subroutine
{% endfor %}
{% endfor %}
{% endfor %}
{%- endmacro -%}
{# #}
{# #}
{# #}
{%- macro gpufort_array_fortran_init_routines(datatypes,max_rank) -%}
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
    mapped_array,&
    bytes_per_element,&
    sizes,lbounds,&
    {{"stream," if is_async}}alloc_mode,sync_mode) result(ierr)
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(inout)        :: mapped_array
  integer(c_int){{size_dims}},intent(in)          :: sizes
  integer(c_int){{size_dims}},intent(in),optional :: lbounds{{"
  type(c_ptr),intent(in)                                                 :: stream" if is_async}}
  integer(c_int),intent(in)                                              :: bytes_per_element
  integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),optional :: alloc_mode 
  integer(kind(gpufort_array_sync_none)),intent(in),optional             :: sync_mode 
  !
  integer(kind(hipSuccess)) :: ierr
  !
  integer(c_int),dimension({{rank}})                 :: opt_lbounds
  integer(kind(gpufort_array_wrap_host_wrap_device)) :: opt_alloc_mode 
  integer(kind(gpufort_array_sync_none))             :: opt_sync_mode 
  !
  opt_lbounds    = 1
  opt_alloc_mode = gpufort_array_alloc_host_alloc_device ! allocate both by default
  opt_sync_mode  = gpufort_array_sync_none
  if ( present(lbounds) )    opt_lbounds    = lbounds             
  if ( present(alloc_mode) ) opt_alloc_mode = alloc_mode 
  if ( present(sync_mode) )  opt_sync_mode  = sync_mode 
  ierr = {{binding}}{{async_suffix}}(&
    mapped_array,&
    bytes_per_element,&
    c_null_ptr,c_null_ptr, &
    sizes, opt_lbounds,&
    {{"stream," if is_async}}opt_alloc_mode, opt_sync_mode )
end function
{% for tuple in datatypes %}
function {{binding}}{{async_suffix}}_{{tuple.f_kind}}(&
    mapped_array,&
    data_host,lbounds,&
    {{"stream," if is_async}}data_dev,alloc_mode,sync_mode) result(ierr)
  use iso_c_binding
  use hipfort_enums
  implicit none
  type({{f_array}}),intent(inout)        :: mapped_array
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_host 
  integer(c_int){{size_dims}},intent(in),optional :: lbounds{{"
  type(c_ptr),intent(in)                                                 :: stream" if is_async}}
  type(c_ptr),intent(in),optional                                        :: data_dev
  integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),optional :: alloc_mode 
  integer(kind(gpufort_array_sync_none)),intent(in),optional             :: sync_mode 
  !
  integer(kind(hipSuccess))                                              :: ierr
  !
  integer(c_int),dimension({{rank}})                     :: opt_lbounds
  type(c_ptr)                                        :: opt_data_dev
  integer(kind(gpufort_array_wrap_host_wrap_device)) :: opt_alloc_mode 
  integer(kind(gpufort_array_sync_none))             :: opt_sync_mode 
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
    mapped_array,&
    int({{tuple.bytes}}, c_int),&
    c_loc(data_host),opt_data_dev,&
    shape(data_host), opt_lbounds,&
    {{"stream," if is_async}}opt_alloc_mode, opt_sync_mode)
end function
{% endfor %}{# datatypes #}
{% endfor %}{# async_suffix #}
{% endfor %}{# rank #}
{%- endmacro -%}
{# #}
{# #}
{# #}
{%- macro gpufort_array_fortran_wrap_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set routine = "init" %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% for tuple in datatypes %}
{% for dev in ["","_cptr"] %} 
{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
function {{f_array}}_wrap{{async_suffix}}_{{tuple.f_kind}}{{dev}}(&
    data_host,data_dev,lbounds,&
    {{"stream," if is_async}}sync_mode,ierr) result(mapped_array)
  use iso_c_binding
  use hipfort_enums
  implicit none
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_host 
{% if dev == "_cptr" %}
  type(c_ptr),intent(in) :: data_dev
{% else %}
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_dev 
{% endif %}
  integer(c_int){{size_dims}},intent(in),optional            :: lbounds{{" 
  type(c_ptr),intent(in)                                     :: stream" if is_async}}
  integer(kind(gpufort_array_sync_none)),intent(in),optional :: sync_mode
  integer(kind(hipSuccess)),intent(inout),optional           :: ierr
  !
  type({{f_array}})         :: mapped_array
  !
  integer(c_int),dimension({{rank}})     :: opt_lbounds
  integer(kind(gpufort_array_sync_none)) :: opt_sync_mode 
  integer(kind(hipSuccess))              :: opt_ierr
  !
  opt_lbounds    = 1
  opt_sync_mode  = gpufort_array_sync_none
  if ( present(lbounds) )   opt_lbounds    = lbounds             
  if ( present(sync_mode) ) opt_sync_mode  = sync_mode 
{% if dev == "_cptr" %}
{% set data_dev_arg = "data_dev" %}
{% else %}
{% set data_dev_arg = "c_loc(data_dev)" %}
{% endif %}
  opt_ierr = {{binding}}{{async_suffix}}(&
    mapped_array,&
    int({{tuple.bytes}}, c_int),&
    c_loc(data_host),{{data_dev_arg}},&
    shape(data_host), opt_lbounds {{",stream" if is_async}},&
    gpufort_array_wrap_host_wrap_device,&
    opt_sync_mode)
  if ( present(ierr) ) ierr = opt_ierr
end function
{% endfor %}{# async #}
{% endfor %}
{% endfor %}
{% set routine = "init" %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
{# function {{f_array}}_wrap_device_ptr_cptr(& #}
{#     data_dev,& #}
{#     sizes,lbounds,& #}
{#     bytes_per_element,ierr) result(mapped_array) #}
{#   use iso_c_binding #}
{#   use hipfort_enums #}
{#   implicit none #}
{#   type(c_ptr),intent(in)                           :: data_dev  #}
{#   integer(c_int){{size_dims}},intent(in)           :: sizes #}
{#   integer(c_int){{size_dims}},intent(in),optional  :: lbounds #}
{#   integer(c_int),intent(in),optional               :: bytes_per_element #}
{#   integer(kind(hipSuccess)),intent(inout),optional :: ierr #}
{#   ! #}
{#   type({{f_array}}) :: mapped_array #}
{#   ! #}
{#   integer(c_int),dimension({{rank}}) :: opt_lbounds #}
{#   integer(c_int)                     :: opt_bytes_per_element #}
{#   integer(kind(hipSuccess))          :: opt_ierr #}
{#   ! #}
{#   opt_lbounds           = 1 #}
{#   opt_bytes_per_element = -1 #}
{#   if ( present(lbounds) )           opt_lbounds           = lbounds              #}
{#   if ( present(bytes_per_element) ) opt_bytes_per_element = bytes_per_element              #}
{#   opt_ierr = {{binding}}(& #}
{#     mapped_array,& #}
{#     opt_bytes_per_element,& #}
{#     c_null_ptr,data_dev,& #}
{#     sizes, opt_lbounds,& #}
{#     gpufort_array_wrap_host_wrap_device,& #}
{#     gpufort_array_sync_none,c_null_ptr) #}
{#   if ( present(ierr) ) ierr = opt_ierr #}
{# end function #}
{% for tuple in datatypes %}
function {{f_array}}_wrap_device_ptr_{{tuple.f_kind}}(&
    data_dev,lbounds,ierr) result(mapped_array)
  use iso_c_binding
  use hipfort_enums
  implicit none
  {{tuple.f_type}},intent(in),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_dev 
  integer(c_int){{size_dims}},intent(in),optional  :: lbounds
  integer(kind(hipSuccess)),intent(inout),optional :: ierr
  !
  type({{f_array}}) :: mapped_array
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  integer(kind(hipSuccess))          :: opt_ierr
  !
  opt_lbounds    = 1
  if ( present(lbounds) )   opt_lbounds    = lbounds             
  opt_ierr = {{binding}}(&
    mapped_array,&
    int({{tuple.bytes}}, c_int),&
    c_null_ptr,c_loc(data_dev),&
    shape(data_dev), opt_lbounds,&
    gpufort_array_wrap_host_wrap_device,&
    gpufort_array_sync_none)
  if ( present(ierr) ) ierr = opt_ierr
end function
{% endfor %}
{% endfor %}
{# #}
{# #}
{# #}
{%- endmacro -%}
{%- macro gpufort_array_fortran_interfaces(datatypes,max_rank) -%}
{% set max_rank_ub = max_rank+1 %}
{% set prefix = "gpufort_array" %}
{% set routine = "init" %}
{% set iface = prefix+"_"+routine %}
!> Initialize a gpufort_array of a rank that matches that of the input data.
{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
interface {{iface}}{{async_suffix}}
{% for rank in range(1,max_rank_ub) %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}} (&
      mapped_array,&
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
    type({{f_array}}),intent(inout)                                     :: mapped_array
    integer(c_int),intent(in),value                                     :: bytes_per_element
    type(c_ptr),intent(in),value                                        :: data_host, data_dev
    integer(c_int){{size_dims}},intent(in)                              :: sizes, lbounds
    integer(kind(gpufort_array_wrap_host_wrap_device)),intent(in),value :: alloc_mode 
    integer(kind(gpufort_array_sync_none)),intent(in),value             :: sync_mode{{"
    type(c_ptr),intent(in),value                                        :: stream" if is_async}}
    !
    integer(kind(hipSuccess)) :: ierr
  end function
{% endfor %}
    module procedure :: &
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
      {{binding}}{{async_suffix}}_host,&
{% for tuple in datatypes %}
      {{binding}}{{async_suffix}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %}
end interface
{% endfor %}{# async_suffix #}

{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
{% set routine = "wrap" %}
{% set iface = prefix+"_"+routine %}
!> Wrap a Fortran or type(c_ptr) pointer pair that points to host and device data.
!> Only wrap, allocate nothing,
!> and do not synchronize at initialization or destruction time.
!> \return gpufort_array of a rank that matches that of the input data.
interface {{iface}}{{async_suffix}}
    module procedure :: &{{"\n"}}
{%- for rank in range(1,max_rank_ub) -%}
{%- set size_dims = ",dimension("+rank|string+")" -%}
{%- set f_array  = prefix+rank|string -%}
{%- set binding  = f_array+"_"+routine -%}
{%- for tuple in datatypes -%}
{%- for dev in ["","_cptr"] -%} 
{{binding | indent(6,True)}}{{async_suffix}}_{{tuple.f_kind}}{{dev}}{{",&\n" if not loop.last}}
{%- endfor -%}{{",&\n" if not loop.last}}
{%- endfor -%}{{",&\n" if not loop.last}}
{%- endfor %}{{""}}
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
{% set size_dims = ",dimension("+rank|string+")" %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
{#      {{binding}}_cptr,& #}
{% for tuple in datatypes %}
      {{binding}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %}
end interface

{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
{% for routine in ["destroy","copy_to_host","copy_to_device"] %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}{{async_suffix}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}} (mapped_array{{",stream" if is_async}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array{{"
    type(c_ptr),value,intent(in)    :: stream" if is_async}}
    integer(kind(hipSuccess))       :: ierr
  end function
{% endfor %}
end interface
{% endfor %}

{% set routine = "dec_num_refs" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}{{async_suffix}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}}{{async_suffix}}(&
    mapped_array,destroy_if_zero_refs{{",&
    stream" if is_async}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array{{"
    type(c_ptr),value,intent(in)               :: stream" if is_async}}
    logical(c_bool),intent(in),value           :: destroy_if_zero_refs
    integer(kind(hipSuccess))                  :: ierr
  end function
{% endfor %}
end interface
{% endfor %}{# async_suffix #}

{% set routine = "inc_num_refs" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}} (mapped_array) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array
    integer(kind(hipSuccess))                  :: ierr
  end function
{% endfor %}
end interface

{%- endmacro -%}
