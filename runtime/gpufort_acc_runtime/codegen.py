#!/usr/bin/env python3
import os
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

    datatypes  =  [\
            ["l","1","logical"], \
            ["i4", "4", "integer(4)"] ,["i8","8","integer(8)"], \
            ["r4","4","real(4)"], ["r8","8","real(8)"], \
            ["c4","2*4","complex(4)"],["c8","2*8","complex(8)"] \
    ]
    dimensions = range(0,max_dims+1)
    context = { "datatypes" : datatypes, "dimensions" : dimensions }
 
    Model(template_path).\
       generate_file(outfile_path,context)
