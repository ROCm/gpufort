{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_param_decl(tavar,
                            qualifiers="",
                            array_and_type_suffix="",
                            interop_suffix="_interop") -%}
{% set name = tavar.c_name if tavar.c_name is defined else tavar.name %}
{%- if tavar.f_type in ["integer","real","logical","type"] -%}
{%-   if tavar.rank > 0 -%}
type(gpufort_array{{tavar.rank}}){{qualifiers}} :: {{name}}
{%-   else -%}
{%-     if tavar.f_type == "type" -%}
{{tavar.f_type}}({{tavar.kind}}{{interop_suffix}}){{qualifiers}} :: {{name}}
{%-     elif tavar.kind is not none and tavar.kind|length -%}
{{tavar.f_type}}({{tavar.kind}}){{qualifiers}} :: {{name}}
{%-     else -%}
{{tavar.f_type}}{{qualifiers}} :: {{name}}
{%-     endif -%}
{%-   endif -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(tavars,
                             qualifiers="",
                             array_and_type_suffix="",
                             interop_suffix="_interop",
                             sep="\n") -%}
{%- for tavar in tavars -%}{{render_param_decl(tavar,qualifiers,array_and_type_suffix,interop_suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(tavars,
                        sep=",") -%}
{%- for tavar in tavars -%}{{tavar.c_name}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_used_modules(used_modules) -%}
{% for module in used_modules %}
{%   if module.renamings is defined and module.renamings|length %}
use {{module.name}},&
{%     for item in module.renamings %}
{%       if item.renamed != item.original %}
{{ (item.renamed + " => " + item.original) | indent(2,True) }}{{",&" if not loop.last}}
{%       else %}
{{ (item.original) | indent(2,True) }}{{",&" if not loop.last}}
{%       endif %}
{%     endfor %}
{%   elif module.only is defined and module.only|length %}
use {{module.name}}, only:&
{%     for item in module.only %}
{%       if item.renamed != item.original %}
{{ (item.renamed + " => " + item.original) | indent(2,True) }}{{",&" if not loop.last}}
{%       else %}
{{ (item.original) | indent(2,True) }}{{",&" if not loop.last}}
{%       endif %}
{%     endfor %}
{%   else %}
use {{module.name}}
{%   endif %}
{% else %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_imports(tavars) -%}
{% for tavar in tavars %}
{% if tavar.rank == 0 and tavar.f_type == "type" %}
import {{tavar.kind}}
{% endif %}{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_set_fptr_lower_bound(fptr,
                                     array,
                                     rank) %}
{% set rank_ub=rank+1 %}
{{fptr}}(&
{% for i in range(1,rank_ub) %}
  lbound({{array}},{{i}}):{{ "," if not loop.last else ")" }}&
{% endfor %}
    => {{fptr}}
{%- endmacro  -%}
{#######################################################################################}