{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_global_param_decl(ivar,taken_by_kernel=False) -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{%- set suffix = "" if taken_by_kernel else "&" -%}
{%- if ivar.rank > 0 -%}
gpufort::array{{ivar.rank}}<{{c_type}}>{{suffix}} {{ivar.name}}
{%- else -%}
{{c_type}}{{suffix}} {{ivar.name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_param_decls(ivars,sep,taken_by_kernel=False) -%}
{%- for ivar in ivars -%}{{render_global_param_decl(ivar,taken_by_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decl(rvar,taken_by_kernel=False) -%}
{%- set c_type = rvar.kind if rvar.f_type=="type" else rvar.c_type -%}
{%- if taken_by_kernel -%}
gpufort::array{{rvar.rank+1}}<{{c_type}}> {{rvar.name}}
{%- else -%}
{{c_type}}& {{rvar.name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decls(rvars,sep,taken_by_kernel=False) -%}
{%- for rvar in rvars -%}{{render_global_reduced_param_decl(rvar,taken_by_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_param_decls(ivars,rvars,taken_by_kernel=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_param_decls(ivars,",\n",taken_by_kernel)}}{{ ",\n" if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_param_decls(rvars,"\n",taken_by_kernel) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_params(ivars) -%}
{%- for ivar in ivars -%}{{ivar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_params(rvars,taken_by_kernel=False) -%}
{%- if taken_by_kernel -%}
{%- for rvar in rvars -%}{%- set rvar_buffer = rvar.name + "_buf" -%}{{rvar_buffer}}{{"," if not loop.last}}{% endfor -%}
{%- else -%}
{%- for rvar in rvars -%}{{rvar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_params(ivars,rvars,taken_by_kernel=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_params(ivars)}}{{ "," if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_params(rvars,taken_by_kernel) }}
{%- endif -%}
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
{# todo must be modified for arrays #}
{%- for ivar in ivars -%}{{render_local_var_decl(ivar)}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_types(types) -%}
{% for derived_type in types %}
typedef struct {
{{ ( render_global_param_decls(derived_type.variables,";\n",True)+";") | indent(2,True) }}
} {{derived_type.name}};{{"\n" if not loop.last }}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_reductions_prepare(rvars) -%}
{%- if rvars|length > 0 -%}
{%- for rvar in rvars %} 
{%- set rvar_buffer = rvar.name + "_buf" -%}
gpufort::array<{{rvar.c_type}}> {{rvar_buffer}};
{{rvar_buffer}}.init(sizeof({{rvar.c_type}})/*bytes per element*/,nullptr/*hostptr*/,nullptr/*deviceptr*/,
  __total_threads(grid,block)/*#elements*/,0/*lower bound*/,AllocMode::WrapHostAllocDevice);
{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_reductions_finalize(rvars) -%}
{%- for rvar in rvars %} 
{%- set rvar_buffer = rvar.name + "_buf" -%}
reduce<{{rvar.c_type}}, reduce_op_{{rvar.op}}>({{rvar_buffer}}.data.data_dev,{{rvar_buffer}}.data.num_elements, {{rvar.name}});
ierr = {{rvar.name}}_buf.destroy();
if ( ierr != hipSuccess ) return ierr;
{%- endfor -%}
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
{%- macro render_launcher_debug_output(prefix,stage,kernel_name,global_vars,global_reduced_vars) -%}
{# Print kernel arguments and array device data (if specified) by runtime
   :param prefix: prefix for every line written out
   :param stage: "INPUT" or "OUTPUT" stage.
   :param kernel_name: Name of the kernel for naming compiler flags
   :param global_vars: list of index variables #}
{% set direction = "in" if stage == "INPUT" else "out" %}
#if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_{{kernel_name}})
std::cout << "{{prefix}}:args:{{direction}}:";
GPUFORT_PRINT_ARGS(grid.x,grid.y,grid.z,block.x,block.y,block.z,sharedmem,stream,{{render_global_params(global_vars+global_reduced_vars)}});
#endif
{% for ivar in global_vars %}
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
{%- macro render_hip_kernel_comment(fortran_snippet) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{fortran_snippet | indent(3, True)}}
*/
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_kernel(kernel) -%}
__global__ void {{kernel.launch_bounds}} {{kernel.name}}(
{{ render_all_global_param_decls(kernel.global_vars,kernel.global_reduced_vars,True) | indent(4,True) }}
){
{% if kernel.local_vars|length %}
{{ render_local_var_decls(kernel.local_vars) | indent(2,True) }}
{% endif %}
{% if kernel.shared_vars|length %}
{{ render_shared_var_decls(kernel.shared_vars) | indent(2,True) }}
{% endif %}
{{kernel.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_device_routine(routine) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{routine.f_body | indent(3, True)}}
*/
__device__ {{routine.return_type}} {{routine.name}}(
{{ render_global_param_decls(routine.global_vars,"","",",\n") | indent(4,True) }}
){
{{ render_local_var_decls(routine.local_vars) | indent(2,True) }}
{{ render_shared_var_decls(routine.shared_vars) | indent(2,True) }}
{{routine.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_kernel_synchronization(kernel_name) -%}
#if defined(SYNCHRONIZE_ALL) || defined(SYNCHRONIZE_{{kernel_name}})
HIP_CHECK(hipStreamSynchronize(stream));
#elif defined(SYNCHRONIZE_DEVICE_ALL) || defined(SYNCHRONIZE_DEVICE_{{kernel_name}})
HIP_CHECK(hipDeviceSynchronize());
#endif
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_launcher(kernel,launcher) -%}
{% set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length %}
extern "C" hipError_t {{launcher.name}}(
{% if launcher.kind == "hip" %}
    dim3& grid,
    dim3& block,
{% endif %}
    const int& sharedmem,
    hipStream_t& stream{{"," if num_global_vars > 0}}
{{ render_all_global_param_decls(kernel.global_vars,kernel.global_reduced_vars) | indent(4,True) }}) {
{% if launcher.kind == "hip_auto" %}
{{ render_block(kernel.name+"_",launcher.block) | indent(2,True) }}
{{ render_grid(kernel.name+"_",launcher.block,launcher.grid,launcher.problem_size) | indent(2,True) }}   
{% endif %}
{% if launcher.debug_output %}
{{ render_launcher_debug_output(kernel.name+":gpu:in:","INPUT",kernel.name,kernel.global_vars,kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
{% if kernel.global_reduced_vars|length %}
{{ render_reductions_prepare(kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  hipLaunchKernelGGL(({{kernel.name}}), grid, block, sharedmem, stream, {{render_all_global_params(kernel.global_vars,kernel.global_reduced_vars,True)}});
{{ render_hip_kernel_synchronization(kernel.name) | indent(2,True) }}
  hipError_t ierr = hipGetLastError();	
  if ( ierr != hipSuccess ) return ierr;
{% if kernel.global_reduced_vars|length %}
{{ render_reductions_finalize(kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
{% if launcher.debug_output %}
{{ render_launcher_debug_output(kernel.name+":gpu:out:","OUTPUT",kernel.name,kernel.global_vars,kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  return hipSuccess;
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_routine(kernel) -%}
{% set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length %}
extern "C" hipError_t {{kernel.name}}_cpu(
    const int& sharedmem,
    const hipStream_t& stream{{"," if num_global_vars > 0}}
{{ render_all_global_param_decls(kernel.global_vars,kernel.global_reduced_vars) | indent(4,True) }});
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_launcher(kernel,launcher) -%}
{% set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length %}
extern "C" hipError_t {{launcher.name}}(
    const int& sharedmem,
    const hipStream_t& stream{{"," if num_global_vars > 0}}
{{ render_all_global_param_decls(kernel.global_vars,kernel.global_reduced_vars) | indent(4,True) }}) {
  hipError_t ierr = hipSuccess;
{% if launcher.debug_output %}
{{ render_launcher_debug_output(kernel.name+":cpu:in:","INPUT",kernel.name,kernel.global_vars,kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  {{kernel.name}}_cpu(sharedmem,stream{{"," if num_global_vars > 0}}{{render_all_global_params(kernel.global_vars,kernel.global_reduced_vars,False)}});
{% if launcher.debug_output %}
{{ render_launcher_debug_output(kernel.name+":cpu:out:","OUTPUT",kernel.name,kernel.global_vars,kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  return hipSuccess;
}
{%- endmacro -%}
