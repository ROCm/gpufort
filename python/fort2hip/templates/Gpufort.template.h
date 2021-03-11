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
lb{{rank}}+i{{rank}}
{%- else -%}
{%- for c in range(1,rank) -%}n{{c}}{{ "*" if not loop.last }}{%- endfor -%}*(lb{{rank}}+i{{rank}})+{{ linearized_index(rank-1) }}
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
#ifndef _GPUFORT_H_
#define _GPUFORT_H_

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

{% set maxRank = 7 -%}
{%- for rank in range(1,maxRank) -%}
#define GPUFORT_PRINT_ARRAY{{rank}}(prefix,A,{{ print_array_arglist("",rank) }}) gpufort_print_array{{rank}}(std::cout, prefix, #A, A, {{ print_array_arglist("",rank) }})
{% endfor -%}

namespace {
  {% for rank in range(1,maxRank+1) %}
  template<typename T>
  void gpufort_print_array{{rank}}(std::ostream& out, const char* prefix, const char* label, T A[], {{ print_array_arglist("int ",rank) }}) {
    int n = {%- for col in range(1,rank+1) -%}n{{col}}{{ "*" if not loop.last }}{%- endfor -%};
    std::vector<T> A_h(n);
    out << prefix << label << ":" << "\n";
    HIP_CHECK(hipMemcpy(A_h.data(), A, n*sizeof(T), hipMemcpyDeviceToHost));
  {% for col in range(1,rank+1) %}
    for ( int i{{rank+1-col}} = 0; i{{rank+1-col}} < n{{rank+1-col}}; i{{rank+1-col}}++ ) {
  {% endfor %}
      T value = A_h[{{ linearized_index(rank) }}];
      out << prefix << label << "(" << {% for col in range(1,rank+1) -%}(lb{{col}}+i{{col}}) << {{ "\",\" <<" |safe if not loop.last }} {% endfor -%} ") = " << std::setprecision(6) << value << "\n";
    {%+ for col in range(1,rank+1) -%}}{%- endfor %} // for loops
  }{{"\n" if not loop.last}}{% endfor %}
}

// global thread indices for various dimensions
#define __gidx(idx) (threadIdx.idx + blockIdx.idx * blockDim.idx) 
#define __gidx1 __gidx(x)
#define __gidx2 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y))
#define __gidx3 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y) + gridDim.x*blockDim.x*gridDim.y*blockDim.y*__gidx(z))
#define __total_threads(grid,block) ( (grid).x*(grid).y*(grid).z * (block).x*(block).y*(block).z )

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

namespace {
  template <typename I, typename E, typename S> __device__ __forceinline__ bool loop_cond(I idx,E end,S stride) {
    return (stride>0) ? ( idx <= end ) : ( -idx <= -end );     
  }

{% for floatType in ["float", "double"] %}  // make {{floatType}}
{% for type in ["short int",  "unsigned short int",  "unsigned int",  "int",  "long int",  "unsigned long int",  "long long int",  "unsigned long long int",  "signed char",  "unsigned char",  "float",  "double",  "long double"] %} __device__ __forceinline__ {{floatType}} make_{{floatType}}(const {{type}}& a) {
    return static_cast<{{floatType}}>(a);
  }
{% endfor %}{% for type in ["hipFloatComplex", "hipDoubleComplex" ] %} __device__ __forceinline__ {{floatType}} make_{{floatType}}(const {{type}}& a) {
    return static_cast<{{floatType}}>(a.x);
  }
{% endfor %}
{% endfor %} // conjugate complex type
  __device__ __forceinline__ hipFloatComplex conj(const hipFloatComplex& c) {
    return hipConjf(c);
  }
  __device__ __forceinline__ hipDoubleComplex conj(const hipDoubleComplex& z) {
    return hipConj(z);
  }

  __device__ __forceinline__ float copysign(const float& a, const float& b) {
    return copysignf(a, b);
  }

{% for op in ["min","max"] %}
{% for n in range(2,16) %}
  {{ binop(op,n) }}
{% endfor %}
{% endfor %}
} 
#endif // _GPUFORT_H_
