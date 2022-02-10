# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

from gpufort import util
from . import opts
        
LOADER = jinja2.FileSystemLoader(os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                               "templates")))
ENV = jinja2.Environment(loader=LOADER,
                         trim_blocks=True,
                         lstrip_blocks=True,
                         undefined=jinja2.StrictUndefined)
TEMPLATES = {}
    
def generate_code(template_path,context={}):
    global ENV
    global TEMPLATES
    if template_path in TEMPLATES:
        template = TEMPLATES[template_path]
    else:
        template = ENV.get_template(template_path)
        TEMPLATES[template_path] = template
    try:
        return template.render(context)
    except Exception as e:
        util.logging.log_error("fort2x.hip.render","generate_code","could not render template '%s'" % template_path)
        raise e

def generate_file(output_path,
                  template_path,
                  context={}):
    with open(output_path,"w") as output:
        output.write(generate_code(template_path,
                                   context))
    
def render_gpufort_header_file(output_path,
                               context={}):
    generate_file(output_path,
                  "gpufort.template.h",
                  context)

def render_gpufort_reduction_header_file(output_path,
                                         context={}):
    generate_file(output_path,
                  "gpufort_reduction.template.h",
                  context)

def render_gpufort_array_header_file(output_path,
                                     context={}):
    generate_file(output_path,
                  "gpufort_array.template.h",
                  context)

def render_gpufort_array_source_file(output_path,
                                     context={}):
    generate_file(output_path,
                  "gpufort_array.template.cpp",
                  context)

@util.logging.log_entry_and_exit(opts.log_prefix)
def render_gpufort_array_fortran_interfaces_file(output_path,
                                                 context={}):
    generate_file(output_path,
                  "gpufort_array.template.f03",
                  context)

parent_dir = os.path.dirname(__file__)
exec(open(os.path.join(parent_dir,"templates","render.py.in")).read())