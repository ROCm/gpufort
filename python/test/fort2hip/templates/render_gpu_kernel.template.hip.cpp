{% import "fort2hip/templates/hip_implementation.macros.hip.cpp" as hm %}
{{ hm.render_gpu_kernel_comment(kernel) }}
{{ hm.render_gpu_kernel(kernel) }} 
