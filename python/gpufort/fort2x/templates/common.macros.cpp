{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_global_param_decl(ivar,is_device_routine=False,is_kernel=False) -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{%- set suffix = "&" if (not is_device_routine
                        or not is_kernel 
                        and (ivar.rank > 0
                            or ivar.f_type == "type"
                            or "intent(out)" in ivar.qualifiers
                            or "intent(inout)" in ivar.qualifiers)) else "" -%}
{%- if ivar.rank > 0 -%}
gpufort::array{{ivar.rank}}<{{c_type}}>{{suffix}} {{ivar.c_name}}
{%- else -%}
{{c_type}}{{suffix}} {{ivar.c_name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_param_decls(ivars,sep,is_device_routine=False,is_kernel=False) -%}
{%- for ivar in ivars -%}{{render_global_param_decl(ivar,is_device_routine,is_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decl(rvar,is_device_routine=False,is_kernel=False) -%}
{%- set c_type = rvar.kind if rvar.f_type=="type" else rvar.c_type -%}
{%- set suffix = "&" if (not is_device_routine
                        or not is_kernel 
                        and (ivar.rank > 0
                            or ivar.f_type == "type"
                            or "intent(out)" in ivar.qualifiers
                            or "intent(inout)" in ivar.qualifiers)) else "" -%}
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
{%- macro render_all_global_param_decls(ivars,rvars,is_device_routine=False,is_kernel=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_param_decls(ivars,",\n",is_device_routine,is_kernel)}}{{ ",\n" if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_param_decls(rvars,"\n",is_device_routine,is_kernel) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_params(ivars) -%}
{%- for ivar in ivars -%}{{ivar.c_name}}{{"," if not loop.last}}{% endfor -%}
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
{%- macro render_all_global_params(ivars,rvars,is_device_routine=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_params(ivars)}}{{ "," if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_params(rvars,is_device_routine) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decl(ivar) -%}
{%- set prefix = "__shared__ " if "shared" in ivar.qualifiers else "" -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{% if ivar.rank > 0 %}
{#    todo add local C array init here too, that one gets a __shared__ prefix if needed #}
gpufort::array{{ivar.rank}}<{{c_type}}> {{ivar.c_name}};
{% else %}
{{prefix}}{{c_type}} {{ivar.c_name}};
{% endif %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decls(ivars) -%}
{# todo must be modified for arrays #}
{%- for ivar in ivars -%}{{render_local_var_decl(ivar)}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_shared_var_decls(ivars) -%}
{# todo must be modified for arrays #}
{%- for ivar in ivars -%}{{render_local_var_decl(ivar)}}{% endfor -%}
{%- endmacro -%}
