{#- SPDX-License-Identifier: MIT                                                -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{#- Jinja2 template for generating interface modules                            -#}
! This file was generated by gpufort
{{prolog}}
module {{name}}
  use iso_c_binding
  use gpufort_array
{% if types is defined and types|length > 0 %}  type

{% for type in types %}
{{type}}
{% endfor %}
{% endif %}
{% if interfaces is defined and interfaces|length > 0 %}  interface

{% for interface in interfaces %}
{{interface}}
{% endfor %}
{% endif %}
{% if routines is defined and routines|length > 0 %}

contains
{% for routine in routines %}
{{routine}}
{% endfor %}
{% endif %}
end module {{name}}