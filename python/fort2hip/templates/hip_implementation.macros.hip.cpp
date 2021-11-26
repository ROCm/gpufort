{########################################################################################}
{%- macro render_param_decl(ivariable,prefix,suffix) -%}
{% if ivariable.rank > 0 %} 
{{prefix}}gpufort::array{{ivariable.rank}}<{{ivariable.c_type}}>{{suffix}} {{ivariable.name}}
{% else %}
{{prefix}}{{ivariable.c_type}}{{suffix}} {{ivariable.name}}
{% endif %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(ivariables,prefix,suffix,sep) -%}
{%- for ivar in ivariables -%}{{render_param_decl(ivariable,prefix,suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(ivariables) -%}
{%- for ivar in ivariables -%}{{ivariable.name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro reductions_prepare(reduced_variables) -%}
{%- if reduced_variables|length > 0 -%}
{%- for rvar in reduced_variables %} 
{{ rvar.c_type }}* {{ rvar.buffer }};
HIP_CHECK(hipMalloc((void **)&{{ rvar.buffer }}, __total_threads(grid,block) * sizeof({{ rvar.c_type }} )));
{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro reductions_finalize(reduced_variables) -%}
{%- if reduced_variables|length > 0 -%}
{%- for rvar in reduced_variables %} 
reduce<{{ rvar.c_type }}, reduce_op_{{ rvar.op }}>({{ rvar.buffer }}, __total_threads(grid,block), {{ rvar.name }});
HIP_CHECK(hipFree({{ rvar.buffer }}));
{% endfor -%}
{%- endif -%}
{%- endmacro %}
{########################################################################################}
{%- macro render_block(prefix,block) -%}
{% for block_dim in block %}
const int {{prefix}}block_dim_{{loop.index}} = {{block_dim}};
{% endfor %}
dim3 block({%- for -%});
{%- endmacro -%}
{########################################################################################}
{%- macro render_grid(prefix,block,grid,problem_size) -%}
{% for size_dim in problem_size %}  
const int {{prefix}}N_{{loop.index}} = {{size_dim}};
{% endfor %}
{% if grid|length > 0 %}
{% for grid_dim in grid %}
const int {{prefix}}grid_dim_{{loop.index}} = {{grid_dim}};
dim3 grid({% for grid_dim in grid -%}{{prefix}}grid_dim_{{loop.index}}{{ "," if not loop.last }}{%- endfor %});
{% endfor %}
{% else %}
dim3 grid({% for block_dim in block %}
  divideAndRoundUp( {{prefix}}N_{{loop.index}}, {{prefix}}block_{{loop.index}} ){{ "," if not loop.last else ");" }}
{% endfor %}
{% endif %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_kernel_launcher_prolog(kernel_name,launcher_kind,ivariables) -%}
#if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_{{kernel_name}})
std::cout << "{{kernel_name}}:{{kind}}:args:";
GPUFORT_PRINT_ARGS(grid.x,grid.y,grid.z,block.x,block.y,block.z,sharedmem,stream,});
#endif
#if defined(GPUFORT_PRINT_INPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAYS_{{kernel_name}})
{% for array in kernel.input_arrays %}
{{array}}.print_device_data(std::cout,"{{kernel_name}}:{{launcher_kind}}in:",gpufort::PrintMode::PrintValuesAndNorms);
{% endfor %}
#elif defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_{{kernel_name}})
{% for array in kernel.input_arrays %}
{{array}}.print_device_data(std::cout,"{{kernel_name}}:{{launcher_kind}}:in:",gpufort::PrintMode::PrintNorms);
{% endfor %}
#endif
{%- endmacro -%}
{########################################################################################}
{%- macro render_kernel_launcher_epilog(kernel_name,launcher_kind,ivariables) -%}
{{ synchronize(kernel_name) }}
#if defined(GPUFORT_PRINT_OUTPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAYS_{{kernel_name}})
{% for ivar in ivariables %}
{% if ivar.rank > 0 %}
{{ivar}}.print_device_data(std::cout,"{{kernel_name}}:{{launcher_kind}}:out:",gpufort::PrintMode::PrintValuesAndNorms);
{% endif %}
{% endfor %}
#elif defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_{{kernel_name}})
{% for ivar in ivariables %}
{% if ivar.rank > 0 %}
{{ivar}}.print_device_data(std::cout,"{{kernel_name}}:{{launcher_kind}}:out:",gpufort::PrintMode::PrintNorms);
{% endif %}
{% endfor %}
#endif
{%- endmacro -%}
{########################################################################################}
{%- macro render_begin_kernel(kernel) -%}
// BEGIN {{kernel.name}} {{kernel.hash}}
{%- endmacro -%}
{########################################################################################}
{%- macro render_end_kernel(kernel) -%}
// END {{kernel.name}} {{kernel.hash}}
{%- endmacro -%}
{########################################################################################}
{%- macro render_kernel(kernel) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{kernel.f_body | indent(3, True)}}
*/
__global__ void {{kernel.launch_bounds}} {{kernel.name}}(
{{ render_param_decls(kernel.params,"","",",\n") | indent(4,True) }}
){
{{ render_locals(kernel.locals) | indent(2,True) }}
{{kernel.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_device_function(kernel) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{kernel.f_body | indent(3, True)}}
*/
__device__ {{kernel.return_type}} {{kernel.name}}(
{{ render_param_decls(kernel.params,"","",",\n") | indent(4,True) }}
){
{{ render_locals(kernel.locals) | indent(2,True) }}
{{kernel.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_kernel_launcher(kernel,launcher_kind) -%}
extern "C" hipError_t {{kernel.interface_name}}(
{% if launcher_kind != "auto" %}
  dim3& grid,
  dim3& block,
{% endif %}
  const int& sharedmem,
  hipStream_t& stream{{"," if kernel.params|length > 0}}
{{ render_param_decls(kernel.params,"","&",",\n") | indent(4,True) }}
) {
{% if launcher_kind == "auto" %}
{{ render_block(kernel.block) }}
{{ render_grid(kernel.block,kernel.grid,kernel.problem_size) }}   
{% endif %}
{% if kernel.generate_debug_code %}
{{ render_kernel_launcher_prolog(kernel) | indent(2,True) }}
{% endif %}
{{ reductions_prepare(kernel) | indent(2,True) }}
  hipLaunchKernelGGL(({{kernel.name}}), grid, block, sharedmem, stream, {{render_params(kernel.params) | join(",")}});
  hipError_t ierr = hipGetLastError();	
  if ( ierr != hipSuccess ) return ierr;
{{ reductions_finalize(kernel) | indent(2,True) }}
{% if kernel.generate_debug_code %}
{{ render_kernel_launcher_epilog(kernel) | indent(2,True) }} 
{% endif %}
  return hipSuccess;
}
{########################################################################################}
{%- macro render_cpu_kernel_interface(kernel) -%}
extern "C" hipError_t {{kernel.interface_name}}_cpu1(
{{ render_param_decls(kernel.params,"","&",",\n") | indent(4,True) }}
);
{########################################################################################}
{%- macro render_cpu_kernel_launcher(kernel) -%}
extern "C" hipError_t {{kernel.interface_name}}_cpu(
  const int& sharedmem,
  const hipStream_t& stream{{"," if kernel.params|length > 0}}
{{ render_param_decls(kernel.params,"","&",",\n") | indent(4,True) }}
) {
  hipError_t ierr = hipSuccess;
{{ reductions_prepare(kernel) | indent(2,True) }}
{{ render_kernel_launcher_prolog(kernel.name,"cpu",kernel.params) | indent(2,True) }}
  if ( ierr != hipSuccess ) return ierr;
  {{ reductions_finalize(kernel) }}
  {{ render_kernel_launcher_epilog(kernel.name,"cpu",kernel.params) | indent(2,True) }} 
  return hipSuccess;
}
{########################################################################################}
