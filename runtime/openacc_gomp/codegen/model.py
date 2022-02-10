#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

class BaseModel():
    def __init__(self,template):
        self._template = template
    def generate_code(self,context={}):
        LOADER = jinja2.FileSystemLoader(os.path.realpath(os.path.dirname(__file__)))
        ENV    = jinja2.Environment(loader=LOADER, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)

        template = ENV.get_template(self._template)
        try:
            return template.render(context)
        except Exception as e:
            print("ERROR: could not render template '%s'" % self._template, file=sys.stderr)
            raise e
    def generate_file(self,output_file_path,context={}):
        with open(output_file_path, "w") as output:
            output.write(self.generate_code(context))

class Model(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/openacc_gomp.template.f")

if __name__ == "__main__":
    maxDims = 7

    mappings = [
            ["present","GOMP_MAP_FORCE_PRESENT"],
            ["create","GOMP_MAP_ALLOC"],
            ["no_create","GOMP_MAP_ALLOC"],
            ["copy","GOMP_MAP_TOFROM"],
            ["copyin","GOMP_MAP_TO"],
            ["copyout","GOMP_MAP_FROM"],
            ["delete","GOMP_MAP_DELETE"]
    ]

    datatypes  =  [\
            ["l","1","logical"], \
            ["i4", "4", "integer(4)"] ,["i8","8","integer(8)"], \
            ["r4","4","real(4)"], ["r8","8","real(8)"], \
            ["c4","2*4","complex(4)"],["c8","2*8","complex(8)"] \
    ]
    dimensions = range(0,maxDims+1)
    context = { 
            "mappings"   : mappings, 
            "datatypes"  : datatypes,
            "datatypes"  : datatypes,
            "dimensions" : dimensions }
    
    model = Model()
    model.generate_file("openacc_gomp.f90",context)