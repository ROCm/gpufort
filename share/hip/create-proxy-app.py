#!/usr/bin/env python3
# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import copy
import argparse
import pprint
import collections
from pyparsing import *
def parse_command_line_arguments():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create a proxy app from a HIP C++ kernel source that has been generated with GPUFORT')
    parser.add_argument('input', help="the input file(s)", nargs="+", type=argparse.FileType("r"))
    parser.add_argument('-o,--output',dest="output",help="the output file", type=argparse.FileType("w"), default=None)
    parser.add_argument('-N', help="default number of iterations", default=10*3,type=int)
    parser.add_argument('-n', help="default array dimension size", default=10,type=int)
    parser.add_argument('-lb', help="default array dimension lower bound", default=1,type=int)
    args = parser.parse_args()
    return args
cl_args = parse_command_line_arguments()

# parser grammar
ParserElement.setDefaultWhitespaceChars(" \t\n\r")
identifier = pyparsing_common.identifier
LPAR,RPAR,COMMA,SEMICOLON = map(Suppress, "(),;")
VOID,LAUNCH,AUTO,RESTRICT = map(Suppress,["void","launch","auto","__restrict__"])
asterisk = Literal("*")
arg      = OneOrMore(identifier|asterisk) + Optional(RESTRICT) + Optional(Literal("[]"),default="")
args     = delimitedList(arg)
launcher_name = identifier.copy()
launcher = VOID + launcher_name + LPAR + Group(args) + RPAR
# parse actions
launcher_context = []
def parse_arg(tokens):
    name    = tokens[-2]
    type    = "".join(tokens[0:-2]).replace("const","").replace("__restrict__","")
    arg = {}
    arg["pointer"] = False
    if "*" in type:
        arg["pointer"] = True
        type = type.replace("*","")
    arg["name"] = name
    arg["type"] = type 
    arg["dims"] = 0
    return arg
def parse_args(tokens):
    for arg in tokens:
       arg["val"] = 1
       arg["array_bound_variable"] = False
       name = arg["name"]
       for other in tokens:
           if name.startswith(other["name"]+"_n"):
               arg["val"] = cl_args.n
               arg["array_bound_variable"] = True
               other["dims"] += 1
           elif name.startswith(other["name"]+"_lb"):
               arg["val"] = cl_args.lb
               arg["array_bound_variable"] = True
    return tokens.asList()
def parse_launcher(tokens):
    global launcher_context
    name, args = tokens
    launcher = {}
    launcher["name"] = name
    launcher["args"] = args
    launcher_context.append(launcher)
    return tokens
arg.setParseAction(parse_arg)
args.setParseAction(parse_args)
launcher.setParseAction(parse_launcher)

# output
scalar_init_template = """{indent}{type} {var} = ({type}) {val};
"""
#array_host_decl_template  = """{indent}{type}* {var}_h = new {type}[{size}];
#""" 
array_host_decl_template  = """{indent}std::vector<{type}> {var}_h({size});
""" 
array_host_init_template  = """{indent}std::fill(std::begin({var}_h), std::end({var}_h), ({type}) {val});
""" 
#array_host_free_template  = """{indent}delete[] {var}_h;
#""" 
array_host_free_template  = "" 
array_dev_init_template  = """{indent}{type}* {var} = nullptr;
{indent}HIP_CHECK(hipMalloc((void**) &{var}, {size}*sizeof({type})));
{indent}HIP_CHECK(hipMemcpy({var}, {var}_h.data(), {var}_h.size()*sizeof({type}), hipMemcpyHostToDevice));
"""
array_dev_free_template  = """{indent}HIP_CHECK(hipFree({var}));
"""
kernel_launch_template  = """{indent}{name}(0,nullptr,{arg_names});
"""
file_template="""
#include <cstdlib>
#include <algorithm>
#include <vector>
{includes}
int main(int argc, char** argv) {{
  // init
{init}
  for (unsigned int i=0; i<{niter}; i++) {{
    // kernel launcher calls
    #ifdef CALL_CPU_LAUNCHER
{cpu_kernel_calls}
    #else
{gpu_kernel_calls}
    #endif
  }}
  // clean up
{finalize}
  return 0;
}}
""".strip()
indent=" "*2

includes         = ""    
cpu_kernel_calls = "" 
gpu_kernel_calls = "" 
scalar_names     = set()
array_names      = set()
array_bound_variables = []
scalars = []
arrays  = []
for kernels_file in cl_args.input:
    content = kernels_file.read()
    includes += "#include \"{file}\"\n".format(file=kernels_file.name)
    # parse 
    cpu_kernel_calls += "{indent2}// {file}:\n".format(indent2=indent*2,file=kernels_file.name)
    gpu_kernel_calls += "{indent2}// {file}:\n".format(indent2=indent*2,file=kernels_file.name)
    launcher.searchString(content)
    launcher_context = [l for l in launcher_context if "_auto" in l["name"]]
    
    for l in launcher_context:
        for arg in l["args"][2:]: # skip shared_mem and stream arg
            if arg["dims"] == 0:
                if arg["name"] not in scalar_names:
                    if arg["array_bound_variable"]:
                        array_bound_variables.append(arg)
                    else:
                        scalars.append(arg) 
                scalar_names.add(arg["name"])
            else:
                if arg["name"] not in array_names:
                    arrays.append(arg)
                array_names.add(arg["name"])
        arg_names = ",".join([("&" if arg["pointer"] else "") + arg["name"] for arg in l["args"][2:]])
        gpu_kernel_calls += kernel_launch_template.format(name=l["name"],arg_names=arg_names,indent=indent*2)
        cpu_kernel_calls += kernel_launch_template.format(name=l["name"].replace("auto","cpu"),arg_names=arg_names,indent=indent*2)
    #pprint.pprint(scalars)
    #pprint.pprint(arrays)
scalar_declarations     = "{indent}// scalar variables\n".format(indent=indent)
host_array_declarations = "{indent}// host array declaration\n".format(indent=indent)
host_array_inits        = "{indent}// host array init\n".format(indent=indent)
host_array_frees        = "{indent}// host array deletion\n".format(indent=indent)
dev_array_inits         = "{indent}// device array init and declaration\n".format(indent=indent)
dev_array_frees         = "{indent}// device array deletion\n".format(indent=indent)
for arg in scalars + array_bound_variables:
    scalar_declarations += scalar_init_template.format(indent=indent,var=arg["name"],val=arg["val"],type=arg["type"])
for arg in arrays:
    name=arg["name"]
    type=arg["type"]
    size = "*".join([ "{name}_n{i}".format(name=name,i=i) for i in range(1,arg["dims"]+1)])
    host_array_declarations += array_host_decl_template.format(indent=indent,var=name,val="1",type=type,size=size)
    host_array_inits        += array_host_init_template.format(indent=indent,var=name,val="1",type=type,size=size)
    host_array_frees        += array_host_free_template.format(indent=indent,var=name)
    dev_array_inits         += array_dev_init_template.format(indent=indent,var=name,val="1",type=type,size=size)
    dev_array_frees         += array_dev_free_template.format(indent=indent,var=name)
init=""
init += scalar_declarations
init += "{indent}// PUT YOUR SCALAR AND ARRAY BOUNDS INIT DATA HERE\n".format(indent=indent)
init += "{indent}// \n".format(indent=indent)
init += "{indent}// END OF YOUR SCALAR AND ARRAY BOUNDS INIT DATA\n".format(indent=indent)
init += host_array_declarations
init += host_array_inits
init += "{indent}// PUT YOUR ARRAY ELEMENT INIT DATA HERE\n".format(indent=indent)
init += "{indent}// \n".format(indent=indent)
init += "{indent}// END OF YOUR ARRAY ELEMENT INIT DATA\n".format(indent=indent)
init += dev_array_inits
finalize=""
#finalize += host_array_frees
finalize += dev_array_frees
proxy_app = file_template.format(\
        indent=indent,indent2=indent*2,\
        niter=cl_args.N,init=init,finalize=finalize,\
        gpu_kernel_calls=gpu_kernel_calls.rstrip(),\
        cpu_kernel_calls=cpu_kernel_calls.rstrip(),\
        includes=includes)
if cl_args.output is None:
    filename = "TEST." + cl_args.input[0].name
else:
    filename = cl_args.output.name
with open(filename,"w") as out:
    out.write(proxy_app)