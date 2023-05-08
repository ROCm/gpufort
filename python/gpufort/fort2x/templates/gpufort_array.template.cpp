{#- GPUFORT -#}
{#- GPUFORT -#}
{% import "gpufort_array.macros.h" as gam %}
// This file was generated from a template via gpufort --gpufort-create-headers
// C bindings
#include "gpufort_array.h"

{{ gam.render_gpufort_array_c_bindings(datatypes,max_rank) }}
