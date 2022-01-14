{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# jinja2 macros #}
{% macro binop_internal(op,n,result,prefix,print_vars) -%}
{%- if n == 1 -%}
{%- set result = prefix+"a" -%}
{%- elif n == 2 -%}
{%- set left = prefix+"a" -%}
{%- set right = prefix+"b" -%}
{%- if print_vars -%}
{%- set result = left + "," + right -%}
{%- else -%}
{%- set result = op+"(" + left + "," + right + ")" -%}
{%- endif -%}
{% else %}
{%- set left = binop_internal(op,(n+n%2)/2,result,prefix+"a",print_vars) -%}
{%- set right = binop_internal(op,n-(n+n%2)/2,result,prefix+"b",print_vars) -%}
{%- if print_vars -%}
{%- set result = left + "," + right -%}
{%- else -%}
{%- set result = op+"(" + left + "," + right + ")" -%}
{%- endif -%}
{%- endif -%}
{{ result }}
{%- endmacro -%}
{%- macro binop(op,n) -%}
#define {{op}}{{n}}({{ binop_internal(op,n,result,"",1) }}) {{ binop_internal(op,n,result,"",0) }}
{%- endmacro -%}
{%- macro linearized_index(rank) -%}
{%- if rank == 1 -%}
i{{rank}}
{%- else -%}
{%- for c in range(1,rank) -%}n{{c}}{{ "*" if not loop.last }}{%- endfor -%}*i{{rank}}+{{ linearized_index(rank-1) }}
{%- endif -%}
{%- endmacro -%}
{%- macro print_array_arglist(prefix,rank) -%}
{%- for col in range(1,rank+1) -%}
{{prefix}}n{{col}},
{%- endfor %}
{%- for col in range(1,rank+1) -%}
{{prefix}}lb{{col}}{{ "," if not loop.last }}
{%- endfor %}
{%- endmacro -%}
{# template body #}
// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_H_
#define _GPUFORT_H_
#include "hip/hip_complex.h"
#include "hip/math_functions.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <vector>
#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

// global thread indices for various dimensions
#define __gidx(idx) (threadIdx.idx + blockIdx.idx * blockDim.idx) 
#define __gidx1 __gidx(x)
#define __gidx2 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y))
#define __gidx3 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y) + gridDim.x*blockDim.x*gridDim.y*blockDim.y*__gidx(z))
#define __total_threads(grid,block) ( (grid).x*(grid).y*(grid).z * (block).x*(block).y*(block).z )
#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

#define GPUFORT_PRINT_ARGS(...) gpufort_show_args(std::cout, #__VA_ARGS__, __VA_ARGS__)
namespace {
  template<typename H1> std::ostream& gpufort_show_args(std::ostream& out, const char* label, H1&& value) {
    return out << label << "=" << std::forward<H1>(value) << '\n';
  }
  
  template<typename H1, typename ...T> std::ostream& gpufort_show_args(std::ostream& out, const char* label, H1&& value, T&&... rest) {
    const char* pcomma = strchr(label, ',');
    return gpufort_show_args(out.write(label, pcomma - label) << "=" << std::forward<H1>(value) << ',',
          pcomma + 1,
          std::forward<T>(rest)...);
  }
}
{% set max_rank = 7 -%}
{%- for rank in range(1,max_rank+1) -%}
#define GPUFORT_PRINT_ARRAY{{rank}}(prefix,print_values,print_norms,A,{{ print_array_arglist("",rank) }}) gpufort_print_array{{rank}}(std::cout, prefix, print_values, print_norms, #A, A, {{ print_array_arglist("",rank) }})
{% endfor -%}
namespace {
  {% for rank in range(1,max_rank+1) %}
  template<typename T>
  void gpufort_print_array{{rank}}(std::ostream& out, const char* prefix, const bool print_values, const bool print_norms, const char* label, T A[], {{ print_array_arglist("int ",rank) }}) {
    int n = {%- for col in range(1,rank+1) -%}n{{col}}{{ "*" if not loop.last }}{%- endfor -%};
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost)); 
    T min = +std::numeric_limits<T>::max();
    T max = -std::numeric_limits<T>::max();
    T sum = 0;
    T l1  = 0;
    T l2  = 0;
{% for col in range(1,rank+1) %}
    for ( int i{{rank+1-col}} = 0; i{{rank+1-col}} < n{{rank+1-col}}; i{{rank+1-col}}++ ) {
{% endfor %}
      T value = A_h[{{ linearized_index(rank) }}];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
        out << prefix << label << "(" << {% for col in range(1,rank+1) -%}(lb{{col}}+i{{col}}) << {{ "\",\" <<" |safe if not loop.last }} {% endfor -%} ") = " << std::setprecision(6) << value << "\n";
      }
    {%+ for col in range(1,rank+1) -%}}{%- endfor %} // for loops
    if ( print_norms ) {
      out << prefix << label << ":" << "min=" << min << "\n";
      out << prefix << label << ":" << "max=" << max << "\n";
      out << prefix << label << ":" << "sum=" << sum << "\n";
      out << prefix << label << ":" << "l1="  << l1  << "\n";
      out << prefix << label << ":" << "l2="  << std::sqrt(l2) << "\n";
    }
  }{{"\n" if not loop.last}}{% endfor %}
  
  // loop condition
  template <typename I, typename E, typename S> __device__ __forceinline__ bool loop_cond(I idx,E end,S stride) {
    return (stride>0) ? ( idx <= end ) : ( -idx <= -end );     
  }

  // type conversions
{% for float_type in ["float", "double"] %}  // make {{float_type}}
{%- for type in ["short int",  "unsigned short int",  "unsigned int",  "int",  "long int",  "unsigned long int",  "long long int",  "unsigned long long int",  "signed char",  "unsigned char",  "float",  "double",  "long double"] +%}
  __device__ __forceinline__ {{float_type}} make_{{float_type}}(const {{type}} a) {
    return static_cast<{{float_type}}>(a);
  }
{%- endfor %}
{%- for type in ["hipFloatComplex", "hipDoubleComplex" ] +%}
  __device__ __forceinline__ {{float_type}} make_{{float_type}}(const {{type}} a) {
    return static_cast<{{float_type}}>(a.x);
  }
{%- endfor %}
{% endfor %} 
  // math functions 
  __device__ __forceinline__ hipFloatComplex conj(const hipFloatComplex& c) {
    return hipConjf(c);
  }
  __device__ __forceinline__ hipDoubleComplex conj(const hipDoubleComplex& z) {
    return hipConj(z);
  }
  __device__ __forceinline__ int nint(const float a) {
    return (a>0.f) ? static_cast<int>(a+0.5f) : static_cast<int>(a-0.5f);
  }
  __device__ __forceinline__ int nint(const double a) {
    return (a>0.) ? static_cast<int>(a+0.5) : static_cast<int>(a-0.5);
  }
  __device__ __forceinline__ float dim(const float a, const float b) {
    const float diff = a-b;
    return (diff>0.f) ? diff : 0.f;
  }
  __device__ __forceinline__ double dim(const double a, const double b) {
    const double diff = a-b;
    return (diff>0.) ? diff : 0.;
  }
} 

#define sign(a,b) copysign(a,b)

{% for op in ["min","max"] %}
{% for n in range(2,15+1) %}
{{ binop(op,n) }}
{% endfor %}
{% endfor %}
#endif // _GPUFORT_H_
