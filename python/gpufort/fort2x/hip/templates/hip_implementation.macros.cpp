{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{% import "common.macros.cpp" as cm %}
{########################################################################################}
{%- macro render_reductions_prepare(rvars) -%}
{%- if rvars|length > 0 -%}
{%- for rvar in rvars %} 
{%- set rvar_buffer = rvar.c_name + "_buf" -%}
gpufort::array{{rvar.rank+1}}<{{rvar.c_type}}> {{rvar_buffer}};
{{rvar_buffer}}.init(sizeof({{rvar.c_type}})/*bytes per element*/,nullptr/*hostptr*/,nullptr/*deviceptr*/,
  __total_threads(grid,block)/*#elements*/,0/*lower bound*/,gpufort::AllocMode::WrapHostAllocDevice);
{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_reductions_finalize(rvars) -%}
{%- for rvar in rvars %} 
{%- set rvar_buffer = rvar.c_name + "_buf" -%}
reduce<{{rvar.c_type}}, reduce_op_{{rvar.op}}>({{rvar_buffer}}.data.data_dev,{{rvar_buffer}}.data.num_elements, {{rvar.c_name}});
ierr = {{rvar.c_name}}_buf.destroy();
if ( ierr != hipSuccess ) return ierr;
{%- endfor -%}
{%- endmacro %}
{########################################################################################}
{%- macro render_grid() -%}
dim3 grid({% for dim in ["x","y","z"] %}divideAndRoundUp(problem_size.{{dim}},block.{{dim}}){{ ",\n          " if not loop.last else ");" }}{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_launcher_debug_output(prefix,
                                       stage,
                                       kernel_name,
                                       global_vars,
                                       global_reduced_vars,
                                       is_gpu_launcher=False,
                                       have_problem_size=False) -%}
{# Print kernel arguments and array device data (if specified) by runtime
   :param prefix: prefix for every line written out
   :param stage: "INPUT" or "OUTPUT" stage.
   :param kernel_name: Name of the kernel for naming compiler flags
   :param global_vars: list of index variables #}
{% set num_args = global_vars|length + global_reduced_vars|length %}
{% set direction = "in" if stage == "INPUT" else "out" %}
#if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_{{kernel_name}})
std::cout << "{{prefix}}:args:{{direction}}:";
GPUFORT_PRINT_ARGS({% if is_gpu_launcher %}{% if have_problem_size %}problem_size.x,problem_size.y,problem_size.y,{% endif %}grid.x,grid.y,grid.z,block.x,block.y,block.z,sharedmem,stream,async{% endif %}{% if num_args %},{{cm.render_global_params(global_vars+global_reduced_vars)}}{% endif %});
#endif
{% for ivar in global_vars %}
{%   if ivar.rank > 0 and ivar.f_type in ["logical","integer","float"] %}
#if defined(GPUFORT_PRINT_{{stage}}_ARRAY_ALL) || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}})_ALL || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}}_{{ivar.c_name}})
{{ivar.c_name}}.print_device_data(std::cout,"{{prefix}}",gpufort::PrintMode::PrintValuesAndNorms);
#elif defined(GPUFORT_PRINT_{{stage}}_ARRAY_ALL) || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}})_ALL || defined(GPUFORT_PRINT_{{stage}}_ARRAY_{{kernel_name}}_{{ivar.c_name}})
{{ivar.c_name}}.print_device_data(std::cout,"{{prefix}}",gpufort::PrintMode::PrintNorms);
#endif
{%   endif %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_device_routine_comment(fortran_snippet) -%}
/**
   HIP C++ implementation of the function/loop body of:

{{fortran_snippet | indent(3, True)}}
*/
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_device_routine(routine) -%}
{{routine.attribute}} {{routine.return_type}} {{routine.launch_bounds}} {{routine.name}}(
{{ cm.render_all_global_param_decls(routine.global_vars+routine.shared_and_local_array_vars,
                                    routine.global_reduced_vars,
                                    True,routine.attribute=="__global__") | indent(4,True) }}
){
{% if routine.local_vars|length %}
{{ cm.render_local_var_decls(routine.local_vars) | indent(2,True) }}
{% endif %}
{% if routine.shared_vars|length %}
{{ cm.render_shared_var_decls(routine.shared_vars) | indent(2,True) }}
{% endif %}
{{routine.c_body | indent(2, True)}}
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_kernel_synchronization(kernel_name) -%}
if ( 
#if defined(SYNCHRONIZE_ALL) || defined(SYNCHRONIZE_{{kernel_name}})
  true ||
#endif
  !async
) { 
  ierr = hipStreamSynchronize(stream);
  if ( ierr != hipSuccess ) return ierr;
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_hip_launcher(kernel,launcher) -%}
{% set num_args = kernel.global_vars|length + kernel.global_reduced_vars|length + kernel.shared_and_local_array_vars|length %}
{% set have_problem_size = launcher.kind == "hip_ps" %}
extern "C" hipError_t {{launcher.name}}_(
    dim3& {{"problem_size" if have_problem_size else "grid"}},
    dim3& block,
    int& sharedmem,
    hipStream_t& stream,
    bool& async{{"," if num_args > 0}}
{{ cm.render_all_global_param_decls(kernel.global_vars,kernel.global_reduced_vars,) | indent(4,True) }}) {
  hipError_t ierr = hipSuccess;
{% if have_problem_size %}
{{ render_grid() | indent(2,True) }}   
{% endif %}
{% if launcher.debug_code %}
{{ render_launcher_debug_output(kernel.name+":gpu:in:","INPUT",kernel.name,
                                kernel.global_vars+kernel.shared_and_local_array_vars,
                                kernel.global_reduced_vars,True,have_problem_size) | indent(2,True) }}
{% endif %}
{% if kernel.global_reduced_vars|length %}
{{ render_reductions_prepare(kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  hipLaunchKernelGGL(({{kernel.name}}), grid, block, sharedmem, stream, {{
    cm.render_all_global_params(
      kernel.global_vars+kernel.shared_and_local_array_vars,
      kernel.global_reduced_vars,True)}});
{{ render_hip_kernel_synchronization(kernel.name) | indent(2,True) }}
{% if kernel.global_reduced_vars|length %}
{{ render_reductions_finalize(kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
{% if launcher.debug_code %}
{{ render_launcher_debug_output(kernel.name+":gpu:out:","OUTPUT",kernel.name,
                                kernel.global_vars+kernel.shared_and_local_array_vars,
                                kernel.global_reduced_vars,True,have_problem_size) | indent(2,True) }}
{% endif %}
  return ierr;
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_routine(kernel) -%}
{% set num_args = kernel.global_vars|length + kernel.global_reduced_vars|length + kernel.shared_and_local_array_vars|length %}
extern "C" hipError_t {{kernel.name}}_cpu(
{{ cm.render_all_global_param_decls(kernel.global_vars+kernel.shared_and_local_array_vars,
                                    kernel.global_reduced_vars) | indent(4,True) }});
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_launcher(kernel,launcher) -%}
{% set num_args = kernel.global_vars|length + kernel.global_reduced_vars|length + kernel.shared_and_local_array_vars|length %}
extern "C" hipError_t {{launcher.name}}_(
    dim3& dummy1,
    dim3& dummy2,
    int& sharedmem,
    hipStream_t& stream,
    bool& async{{"," if num_args > 0}}
{{ cm.render_all_global_param_decls(kernel.global_vars+kernel.shared_and_local_array_vars,
                                    kernel.global_reduced_vars) | indent(4,True) }}) {
  hipError_t ierr = hipSuccess;
{% if launcher.debug_code %}
{{ render_launcher_debug_output(kernel.name+":cpu:in:","INPUT",kernel.name,
                                kernel.global_vars+kernel.shared_and_local_array_vars,
                                kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  {{kernel.name}}_cpu({{cm.render_all_global_params(kernel.global_vars+kernel.shared_and_local_array_vars,
                                                    kernel.global_reduced_vars,False)}});
{% if launcher.debug_code %}
{{ render_launcher_debug_output(kernel.name+":cpu:out:","OUTPUT",kernel.name,
                                kernel.global_vars+kernel.shared_and_local_array_vars,
                                kernel.global_reduced_vars) | indent(2,True) }}
{% endif %}
  return hipSuccess;
}
{%- endmacro -%}