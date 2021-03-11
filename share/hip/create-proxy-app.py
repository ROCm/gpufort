# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import copy
import argparse
import pprint
import collections
from pyparsing import *

def parseCommandLineArguments():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create a proxy app from a HIP C++ kernel source that has been generated with hipfortify.py.')
    parser.add_argument('input', help="the input file(s)", nargs="+", type=argparse.FileType("r"))
    parser.add_argument('-o,--output',dest="output",help="the output file", type=argparse.FileType("w"), default=None)
    parser.add_argument('-N', help="default number of iterations", default=10*3,type=int)
    parser.add_argument('-n', help="default array dimension size", default=10,type=int)
    parser.add_argument('-lb', help="default array dimension lower bound", default=1,type=int)
    args = parser.parse_args()
    return args

clArgs = parseCommandLineArguments()

# parser grammar
ParserElement.setDefaultWhitespaceChars(" \t\n\r")

identifier = pyparsing_common.identifier

LPAR,RPAR,COMMA,SEMICOLON = map(Suppress, "(),;")
VOID,LAUNCH,AUTO          = map(Suppress,["void","launch","auto"])
asterisk = Literal("*")

arg      = OneOrMore(identifier|asterisk) + Optional(Literal("[]"),default="")
args     = delimitedList(arg)
launcherName = identifier.copy()
launcher = VOID + launcherName + LPAR + Group(args) + RPAR

# parse actions
launcherContext = []
def parseArg(tokens):
    name    = tokens[-2]
    type    = "".join(tokens[0:-2]).replace("const","")
    arg = {}
    arg["pointer"] = False
    if "*" in type:
        arg["pointer"] = True
        type = type.replace("*","")
    arg["name"] = name
    arg["type"] = type 
    arg["dims"] = 0
    return arg

def parseArgs(tokens):
    for arg in tokens:
       arg["val"] = 1
       arg["array_bound_variable"] = False
       name = arg["name"]
       for other in tokens:
           if name.startswith(other["name"]+"_n"):
               arg["val"] = clArgs.n
               arg["array_bound_variable"] = True
               other["dims"] += 1
           elif name.startswith(other["name"]+"_lb"):
               arg["val"] = clArgs.lb
               arg["array_bound_variable"] = True
    return tokens.asList()

def parseLauncher(tokens):
    global launcherContext
    name, args = tokens
    launcher = {}
    launcher["name"] = name
    launcher["args"] = args
    launcherContext.append(launcher)
    return tokens

arg.setParseAction(parseArg)
args.setParseAction(parseArgs)
launcher.setParseAction(parseLauncher)

#content="""
#extern "C" void launch_krnl_fe6529_1074_auto(const int sharedMem,
#                                             hipStream_t stream,
#                                             int llb,
#                                             double cgradrot,
#                                             int u_ldown,
#                                             int u_right,
#                                             int ij_end,
#                                             int ij_begin,
#                                             int u_lup,
#                                             double ue_gradrot_e[],
#                                             const int ue_gradrot_e_n1,
#                                             const int ue_gradrot_e_n2,
#                                             const int ue_gradrot_e_lb1,
#                                             const int ue_gradrot_e_lb2,
#                                             int lle) {
#"""
#
#print(launcher.searchString(content))
#pprint.pprint(launcherContext)

scalarInitTemplate = """{indent}{type} {var} = ({type}) {val};
"""
#arrayHostDeclTemplate  = """{indent}{type}* {var}_h = new {type}[{size}];
#""" 
arrayHostDeclTemplate  = """{indent}std::vector<{type}> {var}_h({size});
""" 
arrayHostInitTemplate  = """{indent}std::fill(std::begin({var}_h), std::end({var}_h), ({type}) {val});
""" 
#arrayHostFreeTemplate  = """{indent}delete[] {var}_h;
#""" 
arrayHostFreeTemplate  = "" 
arrayDevInitTemplate  = """{indent}{type}* {var} = nullptr;
{indent}HIP_CHECK(hipMalloc((void**) &{var}, {size}*sizeof({type})));
{indent}HIP_CHECK(hipMemcpy({var}, {var}_h.data(), {var}_h.size()*sizeof({type}), hipMemcpyHostToDevice));
"""
arrayDevFreeTemplate  = """{indent}HIP_CHECK(hipFree({var}));
"""
kernelLaunchTemplate  = """{indent}{name}(0,nullptr,{argNames});
"""

fileTemplate="""
#include <cstdlib>
#include <algorithm>
#include <vector>

{includes}

int main(int argc, char** argv) {{
  // init
{init}

  for (unsigned int i=0; i<{niter}; i++) {{
{body}
  }}

  // clean up
{finalize}
  return 0;
}}
"""

indent=" "*2
kernelCalls = "{indent}// kernel launcher calls\n".format(indent=indent*2) 
includes = ""
    
scalarNames = set()
arrayNames  = set()
arrayBoundVariables = []
scalars = []
arrays  = []

for kernelsFile in clArgs.input:
    content = kernelsFile.read()
    includes += "#include \"{file}\"\n".format(file=kernelsFile.name)
    # parse 
    kernelCalls += "{indent}// {file}:\n".format(indent=indent*2,file=kernelsFile.name)
    launcher.searchString(content)
    launcherContext = [l for l in launcherContext if "_auto" in l["name"]]
    
    for l in launcherContext:
        for arg in l["args"][2:]: # skip sharedMem and stream arg
            if arg["dims"] is 0:
                if arg["name"] not in scalarNames:
                    if arg["array_bound_variable"]:
                        arrayBoundVariables.append(arg)
                    else:
                        scalars.append(arg) 
                scalarNames.add(arg["name"])
            else:
                if arg["name"] not in arrayNames:
                    arrays.append(arg)
                arrayNames.add(arg["name"])
        argNames = ",".join([("&" if arg["pointer"] else "") + arg["name"] for arg in l["args"][2:]])
        kernelCalls += kernelLaunchTemplate.format(name=l["name"],argNames=argNames,indent=indent*2)
    #pprint.pprint(scalars)
    #pprint.pprint(arrays)

scalarDeclarations    = "{indent}// scalar variables\n".format(indent=indent)
hostArrayDeclarations = "{indent}// host array declaration\n".format(indent=indent)
hostArrayInits        = "{indent}// host array init\n".format(indent=indent)
hostArrayFrees        = "{indent}// host array deletion\n".format(indent=indent)
devArrayInits         = "{indent}// device array init and declaration\n".format(indent=indent)
devArrayFrees         = "{indent}// device array deletion\n".format(indent=indent)
for arg in scalars + arrayBoundVariables:
    scalarDeclarations += scalarInitTemplate.format(indent=indent,var=arg["name"],val=arg["val"],type=arg["type"])
for arg in arrays:
    name=arg["name"]
    type=arg["type"]
    size = "*".join([ "{name}_n{i}".format(name=name,i=i) for i in range(1,arg["dims"]+1)])
    hostArrayDeclarations += arrayHostDeclTemplate.format(indent=indent,var=name,val="1",type=type,size=size)
    hostArrayInits        += arrayHostInitTemplate.format(indent=indent,var=name,val="1",type=type,size=size)
    hostArrayFrees        += arrayHostFreeTemplate.format(indent=indent,var=name)
    devArrayInits         += arrayDevInitTemplate.format(indent=indent,var=name,val="1",type=type,size=size)
    devArrayFrees         += arrayDevFreeTemplate.format(indent=indent,var=name)

init=""
init += scalarDeclarations
init += hostArrayDeclarations
init += hostArrayInits
init += devArrayInits

finalize=""
#finalize += hostArrayFrees
finalize += devArrayFrees

body = kernelCalls

proxyApp = fileTemplate.format(niter=clArgs.N,init=init,finalize=finalize,body=body,includes=includes)

if clArgs.output is None:
    filename = "TEST." + clArgs.input[0].name
else:
    filename = clArgs.output.name
with open(filename,"w") as out:
    out.write(proxyApp)
