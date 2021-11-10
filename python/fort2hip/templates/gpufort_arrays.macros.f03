{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{% import "templates/gpufort.macros.h" as gm %}
{%- macro gpufort_arrays_fortran_interfaces(datatypes,max_rank) -%}
{% set main_prefix = "gpufort_mapped_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for tuple in datatypes %}
{% set c_type = tuple.c_type %}
{% set f_type = tuple.f_type %}
interface {{main_prefix}}_{{f_type}}_init_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_init_b (&
      mapped_array,data_host,data_dev,&
{{ gm.separated_list_single_line("n",",",rank) | indent(6,True) }},&
{{ gm.separated_list_single_line("lb",",",rank) | indent(6,True) }},&
      pinned, stream, copyout_at_destruction) &
        bind(c,name="{{c_prefix}}_init") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    type({{f_array}}),intent(inout)  :: mapped_array
    type(c_ptr),intent(in),value     :: data_host, data_dev, stream
    integer(c_int),intent(in),value  :: &
{{ gm.separated_list_single_line("n",",",rank) | indent(6,True) }},&
{{ gm.separated_list_single_line("lb",",",rank) | indent(6,True) }}
    logical(c_bool),intent(in),value :: pinned, copyout_at_destruction
    integer(kind(hipSuccess))        :: ierr
  end function
{% endfor %}
end interface
 
interface {{main_prefix}}_{{f_type}}_destroy_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_destroy(mapped_array) &
     bind(c,name="{{c_prefix}}_destroy") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_copy_data_to_host_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_copy_data_to_host(mapped_array) &
     bind(c,name="{{c_prefix}}_copy_data_to_host") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array_
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_copy_data_to_device_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_copy_data_to_device(mapped_array) &
     bind(c,name="{{c_prefix}}_copy_data_to_device") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_inc_num_refs_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_inc_num_refs(mapped_array) &
     bind(c,name="{{c_prefix}}_inc_num_refs") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_dec_num_refs_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string -%}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type -%}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type -%}
  function {{f_prefix}}_dec_num_refs(mapped_array,destroy_if_zero_refs) &
     bind(c,name="{{c_prefix}}_dec_num_refs") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout)  :: mapped_array
      logical(c_bool),intent(in),value :: destroy_if_zero_refs
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface
{% endfor %}
{%- endmacro -%}
{# type and rank generic interfaces #}
{%- macro gpufort_arrays_fortran_interfaces_generic(datatypes,max_rank) -%}
{% set main_prefix = "gpufort_mapped_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for suffix in ["_init","_destroy","_copy_data_to_host","_copy_data_to_device","_inc_num_refs","_dec_num_refs"] %}
interface {{main_prefix}}{{suffix}}
  module procedure :: &
{% for tuple in datatypes %}
{% set c_type = tuple.c_type %}
{% set f_type = tuple.f_type %}
{% for rank in range(1,max_rank_ub) %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type -%}
{% set routine = f_prefix + suffix %}
{{ routine | indent(4,True) }}{{",&\n" if not loop.last }}
{%- endfor -%}{{",&\n" if not loop.last else "\n" }}   
{%- endfor -%}
end interface
{% endfor %}
{%- endmacro -%}
{# Routines calling the C bindings #}
{%- macro gpufort_arrays_fortran_interfaces(datatypes,max_rank) -%}
{% set main_prefix = "gpufort_mapped_array" %}
{% set max_rank_ub = max_rank+1 %}
{% for tuple in datatypes %}
{% set c_type = tuple.c_type %}
{% set f_type = tuple.f_type %}
interface {{main_prefix}}_{{f_type}}_init_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_init_b (&
      mapped_array,data_host,data_dev,&
{{ gm.separated_list_single_line("n",",",rank) | indent(6,True) }},&
{{ gm.separated_list_single_line("lb",",",rank) | indent(6,True) }},&
      pinned, stream, copyout_at_destruction) &
        bind(c,name="{{c_prefix}}_init") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_array}}
    type({{f_array}}),intent(inout)  :: mapped_array
    type(c_ptr),intent(in),value     :: data_host, data_dev, stream
    integer(c_int),intent(in),value  :: &
{{ gm.separated_list_single_line("n",",",rank) | indent(6,True) }},&
{{ gm.separated_list_single_line("lb",",",rank) | indent(6,True) }}
    logical(c_bool),intent(in),value :: pinned, copyout_at_destruction
    integer(kind(hipSuccess))        :: ierr
  end function
{% endfor %}
end interface
{% endfor %}
 
interface {{main_prefix}}_{{f_type}}_destroy_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_destroy(mapped_array) &
     bind(c,name="{{c_prefix}}_destroy") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_copy_data_to_host_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_copy_data_to_host(mapped_array) &
     bind(c,name="{{c_prefix}}_copy_data_to_host") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array_
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_copy_data_to_device_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_copy_data_to_device(mapped_array) &
     bind(c,name="{{c_prefix}}_copy_data_to_device") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_inc_num_refs_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string %}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type %}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type %}
  function {{f_prefix}}_inc_num_refs(mapped_array) &
     bind(c,name="{{c_prefix}}_inc_num_refs") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout) :: mapped_array
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface

interface {{main_prefix}}_{{f_type}}_dec_num_refs_
{% for rank in range(1,max_rank_ub) %}
{% set f_array  = main_prefix+"_"+rank|string -%}
{% set f_prefix = main_prefix+"_"+rank|string+"_"+f_type -%}
{% set c_prefix = main_prefix+"_"+rank|string+"_"+c_type -%}
  function {{f_prefix}}_dec_num_refs(mapped_array,destroy_if_zero_refs) &
     bind(c,name="{{c_prefix}}_dec_num_refs") result(ierr)
      use iso_c_binding
      use hipfort_enums
      import {{f_array}}
      type({{f_array}}),intent(inout)  :: mapped_array
      logical(c_bool),intent(in),value :: destroy_if_zero_refs
      integer(kind(hipSuccess))  :: ierr
  end function
{% endfor %}
end interface
{% endfor %}
{%- endmacro -%}
