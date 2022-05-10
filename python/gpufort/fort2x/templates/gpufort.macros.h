{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro separated_list_single_line(prefix,sep,rank,suffix="") -%}
{% for d in range(1,rank+1) -%}
{{prefix}}{{d}}{{suffix}}{{ sep if not loop.last }}{%- endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro separated_list(prefix,sep,rank,suffix="") -%}
{% for d in range(1,rank+1) -%}
{{prefix}}{{d}}{{suffix}}{{ sep+"\n" if not loop.last }}
{%- endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro arglist(prefix,rank,suffix="") -%}
{{ separated_list(prefix,",",rank,suffix) }}
{%- endmacro -%}
{########################################################################################}
{%- macro bound_args(prefix,rank) -%}
{{ arglist(prefix+"n",rank) }},
{{ arglist(prefix+"lb",rank) }}
{%- endmacro -%}
{########################################################################################}
{%- macro bound_args_single_line(prefix,rank) -%}
{{ separated_list_single_line(prefix+"n",",",rank) }},
{{ separated_list_single_line(prefix+"lb",",",rank) }}
{%- endmacro -%}
{########################################################################################}
{%- macro render_begin_header(guard) -%}
#ifndef {{guard}}
#define {{guard}}
{%- endmacro -%}
{########################################################################################}
{%- macro render_end_header(guard) -%}
#endif // {{guard}}
{%- endmacro -%}
{########################################################################################}
