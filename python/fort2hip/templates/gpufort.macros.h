{%- macro separated_list_single_line(prefix,sep,rank) -%}
{% for d in range(1,rank+1) -%}
{{prefix}}{{d}}{{ sep if not loop.last }}{%- endfor %}
{%- endmacro -%}
{%- macro separated_list(prefix,sep,rank) -%}
{% for d in range(1,rank+1) -%}
{{prefix}}{{d}}{{ sep+"\n" if not loop.last }}
{%- endfor %}
{%- endmacro -%}
{%- macro arglist(prefix,rank) -%}
{{ separated_list(prefix,",",rank) }}
{%- endmacro -%}
{%- macro bound_args(prefix,rank) -%}
{{ arglist(prefix+"n",rank) }},
{{ arglist(prefix+"lb",rank) }}
{%- endmacro -%}
{%- macro bound_args_single_line(prefix,rank) -%}
{{ separated_list_single_line(prefix+"n",",",rank) }},
{{ separated_list_single_line(prefix+"lb",",",rank) }}
{%- endmacro -%}
