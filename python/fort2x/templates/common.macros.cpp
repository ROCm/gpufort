{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_global_param_decl(ivar,taken_by_kernel=False) -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{%- set suffix = "" if taken_by_kernel else "&" -%}
{%- if ivar.rank > 0 -%}
gpufort::array{{ivar.rank}}<{{c_type}}>{{suffix}} {{ivar.name}}
{%- else -%}
{{c_type}}{{suffix}} {{ivar.name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_param_decls(ivars,sep,taken_by_kernel=False) -%}
{%- for ivar in ivars -%}{{render_global_param_decl(ivar,taken_by_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decl(rvar,taken_by_kernel=False) -%}
{%- set c_type = rvar.kind if rvar.f_type=="type" else rvar.c_type -%}
{%- if taken_by_kernel -%}
gpufort::array{{rvar.rank+1}}<{{c_type}}> {{rvar.name}}
{%- else -%}
{{c_type}}& {{rvar.name}}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_param_decls(rvars,sep,taken_by_kernel=False) -%}
{%- for rvar in rvars -%}{{render_global_reduced_param_decl(rvar,taken_by_kernel)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_param_decls(ivars,rvars,taken_by_kernel=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_param_decls(ivars,",\n",taken_by_kernel)}}{{ ",\n" if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_param_decls(rvars,"\n",taken_by_kernel) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_params(ivars) -%}
{%- for ivar in ivars -%}{{ivar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_global_reduced_params(rvars,taken_by_kernel=False) -%}
{%- if taken_by_kernel -%}
{%- for rvar in rvars -%}{%- set rvar_buffer = rvar.name + "_buf" -%}{{rvar_buffer}}{{"," if not loop.last}}{% endfor -%}
{%- else -%}
{%- for rvar in rvars -%}{{rvar.name}}{{"," if not loop.last}}{% endfor -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_all_global_params(ivars,rvars,taken_by_kernel=False) -%}
{%- if ivars|length > 0 -%}
{{ render_global_params(ivars)}}{{ "," if rvars|length }}
{%- endif -%}
{%- if rvars|length > 0 -%}
{{ render_global_reduced_params(rvars,taken_by_kernel) }}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_local_var_decl(ivar) -%}
{%- set prefix = "__shared__ " if "shared" in ivar.qualifiers else "" -%}
{%- set c_type = ivar.kind if ivar.f_type=="type" else ivar.c_type -%}
{%- if ivar.rank > 0 -%}
{#    todo add local C array init here too, that one gets a __shared__ prefix if needed #}
gpufort::array{{ivar.rank}}<{{c_type}}> {{ivar.name}};
{%- else -%}
{{prefix}}{{c_type}} {{ivar.name}};
{%- endif -%}
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
