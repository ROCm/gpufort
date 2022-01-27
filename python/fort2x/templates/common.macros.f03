{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_param_decl(ivar,
                            qualifiers="",
                            array_and_type_suffix="",
                            interop_suffix="_interop") -%}
{%- if ivar.f_type in ["integer","real","logical","type"] -%}
{%-   if ivar.rank > 0 -%}
type(gpufort_array{{ivar.rank}}){{qualifiers}} :: {{ivar.name}}
{%-   else -%}
{%-     if ivar.f_type == "type" -%}
{{ivar.f_type}}({{ivar.kind}}{{interop_suffix}}){{qualifiers}} :: {{ivar.name}}
{%-     elif ivar.kind is not none and ivar.kind|length -%}
{{ivar.f_type}}({{ivar.kind}}){{qualifiers}} :: {{ivar.name}}
{%-     else -%}
{{ivar.f_type}}{{qualifiers}} :: {{ivar.name}}
{%-     endif -%}
{%-   endif -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(ivars,
                             qualifiers="",
                             array_and_type_suffix="",
                             interop_suffix="_interop",
                             sep="\n") -%}
{%- for ivar in ivars -%}{{render_param_decl(ivar,qualifiers,array_and_type_suffix,interop_suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(ivars,
                        sep=",") -%}
{%- for ivar in ivars -%}{{ivar.name}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_used_modules(used_modules) -%}
{% for module in used_modules %}
use {{module.name}}{{(", only: "+module.only|join(",") if module.only|length>0)}}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
