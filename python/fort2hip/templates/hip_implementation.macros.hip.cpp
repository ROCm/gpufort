{########################################################################################}
{%- macro render_param_decl(ivar,prefix,suffix) -%}
{%- set c_type = ivar.c_type -%}
{%- if ivar.f_type=="type" -%}
{%- set c_type = ivar.kind -%}
{%- endif -%}
{%- if ivar.rank > 0 -%}
{{prefix}}gpufort::array{{ivar.rank}}<{{c_type}}>{{suffix}} {{ivar.name}}
{%- else -%}
{{prefix}}{{c_type}}{{suffix}} {{ivar.name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(ivars,prefix,suffix,sep) -%}
{%- for ivar in ivars -%}{{render_param_decl(ivar,prefix,suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(ivars) -%}
{%- for ivar in ivars -%}{{ivar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decl(ivar) -%}
{%- set prefix = "__shared__ " if "shared" in ivar.qualifiers else "" -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{%- if ivar.rank > 0 -%}
{#    todo add local C array init here too, that one gets a __shared__ prefix if needed #}
gpufort::array{{ivar.rank}}<{{c_type}}> {{ivar.name}};
{%- else -%}
{{prefix}}{{c_type}} {{ivar.name}};
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decls(ivars) -%}
{# todo must be modified for arrays #}
{%- for ivar in ivars -%}{{render_local_var_decl(ivar)}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_shared_var_decls(ivars) -%}
{# todo must be modified arrays #}
{%- for ivar in ivars -%}{{render_local_var_decl(ivar)}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_types(types) -%}
{% for derived_type in types %}
typedef struct {
{{ ( render_param_decls(derived_type.variables,"","",";\n")+";") | indent(2,True) }}
} {{derived_type.name}};{{"\n" if not loop.last }}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_reductions_prepare(reduced_variables) -%}
{%- if reduced_variables|length > 0 -%}
{%- for rvar in reduced_variables %} 
{{rvar.c_type}}* {{rvar.buffer}};
HIP_CHECK(hipMalloc((void **)&{{rvar.buffer}}, __total_threads(grid,block) * sizeof({{rvar.c_type}} )));
{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_reductions_finalize(reduced_variables) -%}
{%- if reduced_variables|length > 0 -%}
{%- for rvar in reduced_variables %} 
reduce<{{rvar.c_type}}, reduce_op_{{rvar.op}}>({{rvar.buffer}}, __total_threads(grid,block), {{rvar.name}});
HIP_CHECK(hipFree({{rvar.buffer}}));
{% endfor -%}
{%- endif -%}
{%- endmacro %}
{########################################################################################}
{%- macro render_block(prefix,block) -%}
{% for block_dim in block %}
const int {{prefix}}block_dim_{{loop.index}} = {{block_dim}};
{% endfor %}
dim3 block({%- for block_dim in block -%}{{prefix}}block_dim_{{loop.index}}{{", " if not loop.last}}{%- endfor -%});
{%- endmacro -%}
{########################################################################################}
{%- macro render_grid(prefix,block,grid,problem_size) -%}
{% if grid|length > 0 %}
{% for grid_dim in grid %}const int {{prefix}}grid_dim_{{loop.index}} = {{grid_dim}};
{% endfor %}
dim3 grid({% for grid_dim in grid -%}{{prefix}}grid_dim_{{loop.index}}{{ "," if not loop.last }}{%- endfor %});
{% else %}
{% for size_dim in problem_size %}const int {{prefix}}N_{{loop.index}} = {{size_dim}};
{% endfor %}
dim3 grid({% for block_dim in block %}
  divideAndRoundUp( {{prefix}}N_{{loop.index}}, {{prefix}}block_{{loop.index}} ){{ "," if not loop.last else ");" }}
{% endfor %}
{% endif %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_kernel_launcher_debug_output(prefix,stage,kernel_name,ivars) -%}
{# Print kernel arguments and array device data (if specified) by runtime
   :param prefix: prefix for every line written out
   :param stage: "INPUT" or "OUTPUT" stage.
   :param kernel_name: Name of the kernel for naming compiler flags
   :param ivars: list of index variables #}
{% set direction = "in" if stage == "INPUT" else "out" %}
#if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_{{kernel_name}})
std::cout << "{{prefix}}:args:{direction}:";
GPUFORT_PRINT_ARGS(grid.x,grid.y,grid.z,block.x,block.y,block.z,sharedmem,stream,{{render_params(ivars)}});
#endif
{% for ivar in ivars %}
{%   if ivar.rank > 0 and ivar.f_type in ["logical","integer","float"] %}
#if defined(GPUFORT_PRINT_{{stage}}_ARRAY_ALL) || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}})_ALL || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}}_{{ivar.name}})
{{ivar.name}}.print_device_data(std::cout,"{{prefix}}",gpufort::PrintMode::PrintValuesAndNorms);
#elif defined(GPUFORT_PRINT_{{stage}}_ARRAY_ALL) || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}})_ALL || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}}_{{ivar.name}})
{{ivar.name}}.print_device_data(std::cout,"{{prefix}}",gpufort::PrintMode::PrintNorms);
#endif
{%   endif %}
{% endfor %}
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
{%- macro render_hip_kernel_comment(kernel) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{kernel.f_body | indent(3, True)}}
*/
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_kernel(kernel) -%}
__global__ void {{kernel.launch_bounds}} {{kernel.name}}(
{{ render_param_decls(kernel.global_vars,"","",",\n") | indent(4,True) }}
){
{{ render_local_var_decls(kernel.local_vars) | indent(2,True) }}
{{ render_shared_var_decls(kernel.shared_vars) | indent(2,True) }}
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
{{ render_param_decls(kernel.global_vars,"","",",\n") | indent(4,True) }}
){
{{ render_local_var_decls(kernel.local_vars) | indent(2,True) }}
{{ render_shared_var_decls(kernel.shared_vars) | indent(2,True) }}
{{kernel.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{% macro render_hip_kernel_synchronization(kernel_name) %}
#if defined(SYNCHRONIZE_ALL) || defined(SYNCHRONIZE_{{kernel_name}})
HIP_CHECK(hipStreamSynchronize(stream));
#elif defined(SYNCHRONIZE_DEVICE_ALL) || defined(SYNCHRONIZE_DEVICE_{{kernel_name}})
HIP_CHECK(hipDeviceSynchronize());
#endif
{% endmacro %}
{########################################################################################}
{%- macro render_hip_kernel_launcher(kernel,kernel_launcher) -%}
{% set num_args = kernel.global_vars|length + kernel.global_reduced_vars|length %}
extern "C" hipError_t {{kernel_launcher.name}}(
{% if kernel_launcher.kind != "auto" %}
    dim3& grid,
    dim3& block,
{% endif %}
    const int& sharedmem,
    hipStream_t& stream{{"," if num_args > 0}}
{{ render_param_decls(kernel.global_vars,"","&",",\n") | indent(4,True) }}
{{ render_param_decls(kernel.global_reduced_vars,"","&",",\n") | indent(4,True) }}
) {
{% if kernel_launcher.kind == "auto" %}
{{ render_block(kernel.name+"_",kernel_launcher.block) | indent(2,True) }}
{{ render_grid(kernel.name+"_",kernel_launcher.block,kernel_launcher.grid,kernel_launcher.problem_size) | indent(2,True) }}   
{% endif %}
{% if kernel_launcher.debug_output %}
{{ render_kernel_launcher_debug_output(kernel.name+":gpu:in:","INPUT",kernel.name,kernel.global_vars) | indent(2,True) }}
{% endif %}
{{ render_reductions_prepare(kernel.global_reduced_vars) | indent(2,True) }}
  hipLaunchKernelGGL(({{kernel.name}}), grid, block, sharedmem, stream, {{render_params(kernel.global_vars)}});
{{ render_hip_kernel_synchronization(kernel.name) | indent(2,True) }}
  hipError_t ierr = hipGetLastError();	
  if ( ierr != hipSuccess ) return ierr;
{{ render_reductions_finalize(kernel.global_reduced_vars) | indent(2,True) }}
{% if kernel_launcher.debug_output %}
{{ render_kernel_launcher_debug_output(kernel.name+":gpu:out:","OUTPUT",kernel.name,kernel.global_vars) | indent(2,True) }}
{% endif %}
  return hipSuccess;
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_kernel_interface(kernel) -%}
extern "C" hipError_t {{kernel.launcher_name}}_cpu1(
{{ render_param_decls(kernel.global_vars,"","&",",\n") | indent(4,True) }}
);
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_kernel_launcher(kernel) -%}
extern "C" hipError_t {{kernel.launcher_name}}_cpu(
  const int& sharedmem,
  const hipStream_t& stream{{"," if kernel.global_vars|length > 0}}
{{ render_param_decls(kernel.global_vars,"","&",",\n") | indent(4,True) }}
) {
  hipError_t ierr = hipSuccess;
{{ render_reductions_prepare(kernel) | indent(2,True) }}
{% if kernel_launcher.debug_output %}
{{ render_kernel_launcher_debug_output(kernel.name+":cpu:in:","INPUT",kernel.name,kernel.global_vars) | indent(2,True) }}
{% endif %}
  if ( ierr != hipSuccess ) return ierr;
{{ render_reductions_finalize(kernel) }}
{% if kernel_launcher.debug_output %}
{{ render_kernel_launcher_debug_output(kernel.name+":cpu:out:","OUTPUT",kernel.name,kernel.global_vars) | indent(2,True) }}
{% endif %}
  return hipSuccess;
}
{%- endmacro -%}
{########################################################################################}
