{% macro render_variable(ivariable,suffix) %}
{% if ivariable.rank > 0 %} 
gpufort::array{{ivariable.rank}}<{{ivariable.c_type}}>{{suffix}} {{ivariable.name}}
{% else %}
{{ivariable.c_type}}{{suffix}} {{ivariable.name}}
{% endif %}
{% endmacro %}
{##}
{##}
{##}
{% macro render_kernel(kernel) %}
{% set krnl_prefix  = kernel.kernel_name %}
// BEGIN {{krnl_prefix}}
/*
   HIP C++ implementation of the function/loop body of:

{{kernel.f_body | indent(3, True)}}
*/
{{kernel.interface_comment if (kernel.interface_comment|length>0)}}
{{kernel.modifier}} {{kernel.return_type}} {{kernel.launch_bounds}} {{kernel.name}}(
{% for ivar in kernel.kernel_args %}
{{ render_variable(ivar,"") | indent(4,True) }}{{"," if not loop.last else ") {"}}
{% endfor -%}
{% for ivar in kernel.kernel_locals %}
{{ render_variable(ivar,"") | indent(2,True) }}{{";" if not loop.last}}
{% endfor %}
{{kernel.c_body | indent(2, True)}}
}
{% endmacro %}
{##}
{##}
{##}
{% macro reductions_prepare %}
{% endmacro %}
{##}
{##}
{##}
{% macro reductions_prepare %}
{% endmacro %}
{##}
{##}
{##}
{% macro render_kernel_launcher(kernel) %}
extern "C" void {{kernel.interface_name}}(
{% for arg in kernel.interface_args %}
{{ render_variable(kernel,"&") | indent(4,True) }}{{"," if not loop.last else ") {"}}
{% endfor -%}
{{ reductions_prepare(kernel,"*") }}{% if kernel.generate_debug_code %}
  #if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_{{krnl_prefix}})
  std::cout << "{{krnl_prefix}}:gpu:args:";
  GPUFORT_PRINT_ARGS((*grid).x,(*grid).y,(*grid).z,(*block).x,(*block).y,(*block).z,sharedmem,stream,{{kernel.kernel_call_arg_names | join(",")}});
  #endif
  #if defined(GPUFORT_PRINT_INPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAYS_{{krnl_prefix}})
  {% for array in kernel.input_arrays %}
  {{ print_array(krnl_prefix+":gpu","in","true","true",array.name,array.rank) }}
  {% endfor %}
  #elif defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_{{krnl_prefix}})
  {% for array in kernel.input_arrays %}
  {{ print_array(krnl_prefix+":gpu","in","false","true",array.name,array.rank) }}
  {% endfor %}
  #endif{% endif +%}
  // launch kernel
  hipLaunchKernelGGL(({{krnl_prefix}}), *grid, *block, sharedmem, stream, {{kernel.kernel_call_arg_names | join(",")}});
{{ reductions_finalize(kernel,"*") }}
{% if kernel.generate_debug_code %}
  {{ synchronize(krnl_prefix) }}
  #if defined(GPUFORT_PRINT_OUTPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAYS_{{krnl_prefix}})
  {% for array in kernel.output_arrays %}
  {{ print_array(krnl_prefix+":gpu","out","true","true",array.name,array.rank) }}
  {% endfor %}
  #elif defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_{{krnl_prefix}})
  {% for array in kernel.output_arrays %}
  {{ print_array(krnl_prefix+":gpu","out","false","true",array.name,array.rank) }}
  {% endfor %}
  #endif
{% endif %}
