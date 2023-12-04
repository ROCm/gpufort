{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_global_param_decl(tavar,is_device_routine=False,is_kernel=False) -%}
{%- set c_type = tavar.kind if tavar.f_type=="type" else tavar.c_type -%}
{%- set rank = tavar.c_rank if tavar.c_rank is defined else tavar.rank -%}
{%- set name = tavar.c_name if tavar.c_name is defined else tavar.name -%}
{%- set suffix = "&" if (not is_device_routine
                        or (not is_kernel 
                           and (rank > 0
                               or tavar.f_type == "type"
                               or "intent(out)" in tavar.attributes
                               or "intent(inout)" in tavar.attributes))) else "" -%}
{%- if rank > 0 -%}
gpufort::array{{rank}}<{{c_type}}>{{suffix}} {{name}}
{%- else -%}
{{c_type}}{{suffix}} {{name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_param_decls(tavars,sep,is_device_routine=False,is_kernel=False) -%}
{%- for tavar in tavars -%}{{render_global_param_decl(tavar,is_device_routine,is_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decl(rvar,is_device_routine=False,is_kernel=False) -%}
{%- set c_type = rvar.kind if rvar.f_type=="type" else rvar.c_type -%}
{%- set suffix = "&" if (not is_device_routine
                        or not is_kernel 
                        and (tavar.rank > 0
                            or tavar.f_type == "type"
                            or "intent(out)" in tavar.attributes
                            or "intent(inout)" in tavar.attributes)) else "" -%}
{%- if is_device_routine -%}
gpufort::array{{rvar.rank+1}}<{{c_type}}>{{suffix}} {{rvar.c_name}}
{%- else -%}
{{c_type}}{{suffix}} {{rvar.c_name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decls(rvars,sep,is_device_routine=False,is_kernel=False) -%}
{%- for rvar in rvars -%}{{render_global_reduced_param_decl(rvar,is_device_routine,is_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_param_decls(tavars,rvars,is_device_routine=False,is_kernel=False) -%}
{%- if tavars|length -%}
{{ render_global_param_decls(tavars,",\n",is_device_routine,is_kernel)}}{{ ",\n" if rvars|length }}
{%- endif -%}
{%- if rvars|length -%}
{{ render_global_reduced_param_decls(rvars,",\n",is_device_routine,is_kernel) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_params(tavars) -%}
{%- for tavar in tavars -%}{{tavar.c_name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_params(rvars,is_device_routine=False) -%}
{%- if is_device_routine -%}
{%- for rvar in rvars -%}{%- set rvar_buffer = rvar.c_name + "_buf" -%}{{rvar_buffer}}{{"," if not loop.last}}{% endfor -%}
{%- else -%}
{%- for rvar in rvars -%}{{rvar.c_name}}{{"," if not loop.last}}{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_params(tavars,rvars,is_device_routine=False) -%}
{%- if tavars|length > 0 -%}
{{ render_global_params(tavars)}}{{ "," if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_params(rvars,is_device_routine) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decl(tavar) -%}
{%- set prefix = "__shared__ " if "shared" in tavar.attributes else "" -%}
{%- set c_type = tavar.kind if tavar.f_type=="type" else tavar.c_type -%}
{% if tavar.rank > 0 %}
{#    todo add local C array init here too, that one gets a __shared__ prefix if needed #}
{{prefix}}{{c_type}} _{{tavar.c_name}}[{{tavar.size|join("*")}}];
gpufort::array_descr{{tavar.rank}}<{{c_type}}> {{tavar.c_name}};
{{tavar.c_name}}.wrap(nullptr,&_{{tavar.c_name}}[0],
  {{tavar["size"] | join(",")}},
  {{tavar["lbounds"] | join(",")}});
{% else %}
{{prefix}}{{c_type}} {{tavar.c_name}};
{% endif %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decls(tavars) -%}
{# todo must be modified for arrays #}
{%- for tavar in tavars -%}{{render_local_var_decl(tavar)}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_shared_var_decls(tavars) -%}
{# todo must be modified for arrays #}
{%- for tavar in tavars -%}{{render_local_var_decl(tavar)}}{% endfor -%}
{%- endmacro -%}