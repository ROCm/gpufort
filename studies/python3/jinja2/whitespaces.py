#!/usr/bin/env python3
# Make sure to read: https://ttl255.com/jinja2-tutorial-part-3-whitespace-control/
import jinja2

loop1="""\
before1

before2
{% for i in range(1,3) %}
{{i}}
{% endfor %}
after1

after2
"""

def make_prefix(trim_blocks,
                lstrip_blocks,
                keep_trailing_newline):
    return ("trim_blocks" if trim_blocks else "x")\
           + "," + ("lstrip_blocks" if lstrip_blocks else "x")\
           + "," + ("keep_trailing_newline" if keep_trailing_newline else "")
    

for trim in [False,True]:
    for lstrip in [False,True]:
        for keep in [False,True]:
            ENV = jinja2.Environment(loader=jinja2.BaseLoader,
                                     trim_blocks=trim,
                                     lstrip_blocks=lstrip,
                                     keep_trailing_newline=keep,
                                     undefined=jinja2.StrictUndefined)
            #print(make_prefix(trim,lstrip,keep)+":")
            #print("'"+ENV.from_string(loop1).render()+"'")

            if trim and lstrip and not keep:
                # following adds extra newline at the end of print statement
                loop2_v1="""\
{%- macro print() -%}
{% for i in range(1,3) %}
{{i}}
{% endfor %}
{%- endmacro -%}
before1
{{ print() }}
after1
"""
                print("'"+ENV.from_string(loop2_v1).render()+"'")
                # Can be prevented by using '-}}' when rendering the template
                loop2_v2="""\
{%- macro print() -%}
{% for i in range(1,3) %}
{{i}}
{% endfor %}
{%- endmacro -%}
before1
{{ print() -}}
after1
"""
                print("'"+ENV.from_string(loop2_v2).render()+"'")
