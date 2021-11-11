{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{% import "templates/gpufort.macros.h" as gm %}
{%- macro gpufort_arrays_fortran_routines(datatypes,max_rank) -%}
{% set prefix = "gpufort_mapped_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+"_"+rank|string %}
{% set routine = "init" %}
{% set binding  = f_array+"_"+routine %}
{% set size_dims = ",dimension("+rank|string+")" %}
  function {{binding}}_host(&
      mapped_array,&
      bytes_per_element,&
      sizes,lbounds,&
      pinned, copyout_at_destruction) result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(inout)        :: mapped_array
    integer(c_int),intent(in),value            :: bytes_per_element
    integer(c_int){{size_dims}},intent(in)          :: sizes
    integer(c_int){{size_dims}},intent(in),optional :: lbounds
    logical(c_bool),intent(in),optional                :: pinned, copyout_at_destruction
    integer(kind(hipSuccess))                          :: ierr
    !
    integer(c_int),dimension({{rank}}) :: opt_lbounds
    logical(c_bool)                :: opt_pinned, opt_copyout_at_destruction
    !
    opt_lbounds                = 1
    opt_copyout_at_destruction = .false.
    if ( present(lbounds) )                opt_lbounds                = lbounds             
    if ( present(copyout_at_destruction) ) opt_copyout_at_destruction = copyout_at_destruction 
    ierr = {{binding}}(&
      mapped_array,&
      bytes_per_element,&
      c_null_ptr,c_null_ptr, &
      sizes, opt_lbounds,&
      opt_pinned, opt_copyout_at_destruction)
  end function
{% for tuple in datatypes %}
  function {{binding}}_{{tuple.f_kind}}(&
      mapped_array,&
      data_host,lbounds,&
      data_dev,&
      pinned,copyout_at_destruction) result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(inout)        :: mapped_array
    {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: data_host 
    integer(c_int){{size_dims}},intent(in),optional :: lbounds
    type(c_ptr),intent(in),optional                    :: data_dev
    logical(c_bool),intent(in),optional                :: pinned, copyout_at_destruction
    integer(kind(hipSuccess))                          :: ierr
    !
    integer(c_int),dimension({{rank}}) :: opt_lbounds
    type(c_ptr)                    :: opt_data_dev
    logical(c_bool)                :: opt_pinned, opt_copyout_at_destruction
    !
    opt_lbounds                = 1
    opt_data_dev               = c_null_ptr
    opt_copyout_at_destruction = .false.
    if ( present(lbounds) )                opt_lbounds                = lbounds             
    if ( present(data_dev) )               opt_data_dev               = data_dev              
    if ( present(copyout_at_destruction) ) opt_copyout_at_destruction = copyout_at_destruction 
    ierr = {{binding}}(&
      mapped_array,&
      int({{tuple.bytes}}, c_int),&
      c_loc(data_host),opt_data_dev,&
      shape(data_host), opt_lbounds,&
      opt_pinned, opt_copyout_at_destruction)
  end function
  
{% endfor %}
{% endfor %}
{%- endmacro -%}
{%- macro gpufort_arrays_fortran_interfaces(datatypes,max_rank) -%}
{% set max_rank_ub = max_rank+1 %}
{% set prefix = "gpufort_mapped_array" %}
{% set routine = "init" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set size_dims = ",dimension("+rank|string+")" %}
{% set f_array  = prefix+"_"+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}} (&
      mapped_array,&
      bytes_per_element,&
      data_host,data_dev,&
      sizes, lbounds,&
      pinned, copyout_at_destruction) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array
    integer(c_int),intent(in),value            :: bytes_per_element
    type(c_ptr),intent(in),value               :: data_host, data_dev
    integer(c_int){{size_dims}},intent(in)     :: sizes, lbounds
    logical(c_bool),intent(in),value           :: pinned, copyout_at_destruction
    integer(kind(hipSuccess))                  :: ierr
  end function
{% endfor %}
    module procedure :: &
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+"_"+rank|string %}
{% set binding  = f_array+"_"+routine %}
      {{binding}}_host,&
{% for tuple in datatypes %}
      {{binding}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %}
end interface

{% for routine in ["destroy","copy_data_to_host","copy_data_to_device"] %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+"_"+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}} (mapped_array,stream) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array
    type(c_ptr),value,intent(in)    :: stream
    integer(kind(hipSuccess))       :: ierr
  end function
{% endfor %}
end interface
{% endfor %}

{% set routine = "inc_num_refs" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+"_"+rank|string %}
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

{% set routine = "dec_num_refs" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = prefix+"_"+rank|string %}
{% set binding  = f_array+"_"+routine %}
  function {{binding}} (mapped_array,destroy_if_zero_refs) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    implicit none
    type({{f_array}}),intent(inout) :: mapped_array
    logical(c_bool),intent(in),value           :: destroy_if_zero_refs
    integer(kind(hipSuccess))                  :: ierr
  end function
{% endfor %}
end interface
{%- endmacro -%}
