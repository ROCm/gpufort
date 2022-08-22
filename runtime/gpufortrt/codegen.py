#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import pprint
import argparse

import jinja2

class Model():
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

def parse_cl_args():
    parser = argparse.ArgumentParser(description="Codegenerator")
    parser.add_argument("input",type=str,help="Path to template file. (in format: <name>.template.<ext>")
    parser.add_argument("-d","--max-dims",type=int,dest="max_dims",help="Maximum number of array dimensions to support.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cl_args()

    max_dims      = args.max_dims
    template_path = args.input
    outfile_path  = template_path.replace(".template","")

    # suffix, bytes, Fortran type (please double check bytes)
    datatypes  =  [
      ("l1","1","logical(c_bool)"), 
      ("l4","4","logical"), 
      ("ch1", "1", "character(c_char)") ,
      ("i2","2","integer(c_short)"), 
      ("i4", "4", "integer(c_int)") ,
      ("i8","8","integer(c_long)"), 
      ("r4","4","real(c_float)"), 
      ("r8","8","real(c_double)"), 
      ("c4","2*4","complex(c_float_complex)"),
      ("c8","2*8","complex(c_double_complex)"),
    ]
    context = {"datatypes" : datatypes,
               "max_rank" : max_dims}
 
    Model(template_path).\
       generate_file(outfile_path,context)
