{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{% import "common.macros.cpp" as cm %}
{########################################################################################}
{%- macro render_derived_types(types) -%}
{% for derived_type in types %}
typedef struct {
{{ ( cm.render_global_param_decls(derived_type.variables,";\n",True)+";") | indent(2,True) }}
} {{derived_type.name}};{{"\n" if not loop.last }}
{% endfor %}
{%- endmacro -%}
