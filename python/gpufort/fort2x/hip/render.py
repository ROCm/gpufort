# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

from . import opts

from gpufort import util

LOADER = jinja2.FileSystemLoader(
    os.path.realpath(os.path.join(os.path.dirname(__file__), "templates")))
ENV = jinja2.Environment(loader=LOADER,
                         trim_blocks=True,
                         lstrip_blocks=True,
                         undefined=jinja2.StrictUndefined)
TEMPLATES = {}


def generate_code(template_path, context={}):
    global ENV
    global TEMPLATES
    try:
        if template_path in TEMPLATES:
            template = TEMPLATES[template_path]
        else:
            template = ENV.get_template(template_path)
            TEMPLATES[template_path] = template
        return template.render(context)
    except Exception as e:
        raise IOError("could not render template '%s'" % template_path)


def generate_file(output_path, template_path, context={}):
    with open(output_path, "w") as output:
        output.write(generate_code(template_path, context))

parent_dir = os.path.dirname(__file__)
include_file = os.path.abspath(os.path.join(parent_dir, "templates", "render.py.in"))
if os.path.exists(include_file):
    exec(open(include_file).read())
else:
    util.logging.log_warning(opts.log_prefix,"<module load>","file '{}' not found".format(include_file))
