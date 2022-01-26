# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

import addtoplevelpath
import utils.logging
        
LOADER    = jinja2.FileSystemLoader(os.path.realpath(os.path.join(os.path.dirname(__file__),"templates")))
ENV       = jinja2.Environment(loader=LOADER,
                               trim_blocks=True,
                               #lstrip_blocks=True,
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
        utils.logging.log_error("fort2x.hip.render","generate_code","could not render template '%s'" % template_path)
        raise e

def generate_file(output_path,template_path,context={}):
    with open(output_path, "w") as output:
        output.write(generate_code(template_path,context))

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"templates","render.py.in")).read())
