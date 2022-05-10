{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro this_stride(r,array_rank) -%}
{%- if r == 1 -%}
1
{%- elif r == array_rank+1 -%}
this->num_elements
{%- else -%}
this->stride{{r}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_dopes(max_rank) -%}
namespace gpufort {
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  /**
   * Describes meta information of a multi-dimensional array
   * such as the lower and upper bounds per dimension, the extent per dimension, the
   * strides per dimension and the total number of elements.
   * It overloads `operator()` to linearize a multi-dimensional index
   * according to the aforementioned meta information.
   */
  struct dope{{rank}} {
    int index_offset = -1;       //> Offset for index calculation; scalar product of negative lower bounds and strides.
{% for d in range(2,rank_ub) %}
    int stride{{d}} = -1;    //> Stride {{d}} for linearizing {{rank}}-dimensional index.
{% endfor %}
    int num_elements = 0;        //> Number of elements represented by this array.
 
    /**
     * (Re-)Init the dope vector. 
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    GPUFORT_DEVICE_ROUTINE_INLINE void init(
{{ gm.separated_list("const int n",",",rank) | indent(6,True) }},
{{ gm.separated_list("const int lb",",",rank) | indent(6,True) }}
    ) {
       // column-major access
{% for d in range(2,rank_ub+1) %}
       {{ this_stride(d,rank) }} = {{ this_stride(d-1,rank) }}*n{{d-1}};
{% endfor %}
       this->index_offset =
{% for d in range(1,rank_ub) %}
         -lb{{d}}*{{ this_stride(d,rank) }}{{";" if loop.last}}
{% endfor %}
    }
    
    /**
     * (Re-)Init the dope vector. 
     * \param[in] sizes extent per dimension; array with `rank` elements.  
     * \param[in] lbounds lower bound per dimension; array with `rank` elements.
     */ 
    GPUFORT_DEVICE_ROUTINE_INLINE void init(
      const int* const sizes,
      const int* const lbounds,
    ) {
      this->init(
{{ gm.separated_list_single_line("sizes[",",",rank,"]") | indent(8,True) }},
{{ gm.separated_list_single_line("lbounds[",",",rank,"]") | indent(8,True) }}
      );
    }

    /**
     * Copy from another dope vector.
     */
    GPUFORT_DEVICE_ROUTINE_INLINE copy(const dope{{rank}}& other) {
      this->index_offset = other.index_offset;
      {% for d in range(2,rank_ub) %}
      this->stride{{d}} = other.stride{{d}};
      {% endfor %}
      this->num_elements = num_elements;
    }

    /** Constructor. */ 
    GPUFORT_DEVICE_ROUTINE dope{{rank}}(
{{ gm.separated_list("const int n",",",rank) | indent(6,True) }},
{{ gm.separated_list("const int lb",",",rank) | indent(6,True) }}
    ) {
      this->init(
{{ gm.separated_list_single_line("n",",",rank) | indent(8,True) }},
{{ gm.separated_list_single_line("lb",",",rank) | indent(8,True) }}
      ); 
    } 
   
    /** Constructor.
     * \param[in] sizes extent per dimension; array with `rank` elements.  
     * \param[in] lbounds lower bound per dimension; array with `rank` elements.
     */
    GPUFORT_DEVICE_ROUTINE_INLINE dope{{rank}}(
      const int* const sizes,
      const int* const lbounds,
    ) {
      this->init(sizes,lbounds);
    } 
    
    GPUFORT_DEVICE_ROUTINE_INLINE copy(const dope{{rank}}& other) {
      this->index_offset = other.index_offset;
      {% for d in range(2,rank_ub) %}
      this->stride{{d}} = other.stride{{d}};
      {% endfor %}
      this->num_elements = num_elements;
    }
   
    /** Constructor. */ 
    GPUFORT_DEVICE_ROUTINE_INLINE dope{{rank}}(
    ) {}
    
    
    /** Copy constructor. */
    GPUFORT_DEVICE_ROUTINE_INLINE dope(const dope{{rank}}& other) {
      this->copy(other);
    }
 
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    GPUFORT_DEVICE_ROUTINE_INLINE int operator() (
{{ gm.arglist("const int i",rank) | indent(6,True) }}
    ) const {
      return this->index_offset
{% for d in range(1,rank_ub) %}
          + i{{d}}*{{ this_stride(d,rank) }}{{";" if loop.last}}
{% endfor %}
    }
    
    /**
     * \return number of array elements.
     */
    GPUFORT_DEVICE_ROUTINE_INLINE int size() const {
      return {{ this_stride(rank+1,rank) }};
    }
    
    /**
     * \return size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    GPUFORT_DEVICE_ROUTINE_INLINE int size(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= {{rank}});
      #endif
      switch(dim) {
{% for r in range(rank,0,-1) %}
        case {{r}}:
          return {{ this_stride(r+1,rank) }} / {{ this_stride(r,rank) }};
{% endfor %}
        default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array{{rank}}::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: {{rank}}" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    GPUFORT_DEVICE_ROUTINE_INLINE int lbound(int dim) const {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= {{rank}});
      #endif
      int offset_remainder = this->index_offset;
{% for r in range(rank,0,-1) %}
      {{"int " if r == rank}}lb = -offset_remainder / {{ this_stride(r,rank) }};
{% if r == 1 %}
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*{{ this_stride(r,rank) }};
      assert(offset_remainder == 0);
      #endif
      return lb;
{% else %}
      if ( dim == {{r}} ) return lb;
      offset_remainder = offset_remainder + lb*{{ this_stride(r,rank) }};
{% endif %}
{% endfor %}
    }
    
    /**
     * \return upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    GPUFORT_DEVICE_ROUTINE_INLINE int ubound(int dim) const {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  }; // class dope{{rank}}
{% endfor %}
} // namespace gpufort
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_dope_c_bindings(max_rank) -%}
{% for rank in range(1,max_rank+1) %}
{%  set c_prefix = "gpufort_dope"+rank|string %}
GPUFORT_HOST_ROUTINE void {{c_prefix}}_init(
  gpufort::{{c_prefix}}{{rank}}* dope,
  const int* const sizes,
  const int* const lbounds
) {
  dope->init(sizes,lbounds); 
}

/**
 * \return size in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE int {{c_prefix}}_size(
    gpufort::dope{{rank}}* dope,
    const int& dim) {
  return dope->size(dim);
}

/**
 * \return lower bound (inclusive) in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE int {{c_prefix}}_lbound(
    gpufort::dope{{rank}}* dope,
    const int dim) {
  return dope->lbound(dim);
}

/**
 * \return upper bound (inclusive) in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE int {{c_prefix}}_ubound(
    gpufort::dope{{rank}}* dope,
    const int dim) {
  return dope->ubound(dim);
}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
