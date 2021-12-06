{#- SPDX-License-Identifier: MIT                                                -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{#- Jinja2 template for generating interface modules                            -#}
{#- This template works with data structures of the form :                      -#}
{########################################################################################}
{%- macro render_param_decl(ivar,type_suffix) -%}
{%- if ivar.f_type in ["integer","real","logical","type"] -%}
{%- if ivar.rank > 0 -%}
type(gpufort_array{{ivar.rank}}) :: {{ivar.name}}
{%- else -%}
{%- if ivar.f_type == "type" -%}
{{ivar.f_type}}({{ivar.kind}}{{type_suffix}}) :: {{ivar.name}}
{%- elif ivar.kind is not none and ivar.kind|length > 0 -%}
{{ivar.f_type}}({{ivar.kind}}) :: {{ivar.name}}
{%- else -%}
{{ivar.f_type}} :: {{ivar.name}}
{%- endif -%}
{%- endif -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(ivars,type_suffix,sep) -%}
{%- for ivar in ivars -%}{{render_param_decl(ivar,type_suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(ivars) -%}
{%- for ivar in ivars -%}{{ivar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_types(types,type_suffix) -%}
{% for derived_type in types %}
type, bind(c) :: {{derived_type.name}}{{type_suffix}}
{{ (render_param_decls(derived_type.variables,type_suffix,"\n")) | indent(2,True) }}
end type {{derived_type.name}}{{type_suffix}}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
