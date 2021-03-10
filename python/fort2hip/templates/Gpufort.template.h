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
{%- endmacro %}

{%- macro binop(op,n) -%}
#define {{op}}{{n}}({{ binop_internal(op,n,result,"",1) }}) {{ binop_internal(op,n,result,"",0) }}
{%- endmacro %}

{# template body #}
#ifndef _GPUFORT_H_
#define _GPUFORT_H_

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

  __device__ __forceinline__ float copysign(const floatA, const floatB) {
    return copysignf(floatA, floatB);
  }

{% for op in ["min","max"] %}
{% for n in range(2,16) %}
  {{ binop(op,n) }}
{% endfor %}
{% endfor %}
} 
#endif
