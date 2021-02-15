#!/usr/bin/env python3
from pyparsing import *

ParserElement.setDefaultWhitespaceChars(" \t\n\r")

LPAR,RPAR,COMMA,SEMICOLON = map(Suppress, "(),;")
BRA,KET = map(Suppress,["<<<",">>>"])

arg  = Combine(Optional("*") + pyparsing_common.identifier)
args = delimitedList(arg) 
kernelLaunch = Combine(ZeroOrMore(White())) +  pyparsing_common.identifier + BRA + args + KET + LPAR + args + RPAR + SEMICOLON

content="""
  // launch kernel
  krnl_1fb5dc_165<<<
                    *grid,
                     *block,
                     sharedMem,
                     stream>>>(
                     ij_omp_begin_ext,
                     ij_omp_end_ext,
                     l,
                     rhodz,
                     rhodz_n1,
                     rhodz_n2,
                     rhodz_lb1,
                     rhodz_lb2,
                     g,
                     geopot,
                     geopot_n1,
                     geopot_n2,
                     geopot_lb1,
                     geopot_lb2);
"""

def parseAction(tk):
    indent = tk[0].replace("\n","")
    return """\n{0}hipLaunchKernelGGL(({1}),{2},{3},{4},{5},
{0}  {6}
{0});
""".format(\
            indent,tk[1],tk[2],tk[3],tk[4],tk[5],
            (",\n{0}  ".format(indent)).join(tk[6:]))

kernelLaunch.setParseAction(parseAction)
print(kernelLaunch.transformString(content))

import sys
sys.exit()

files="""
caldyn_kernels_base.kernels.hip.cpp
caldyn_kernels_hevi.kernels.hip.cpp
caldyn_hevi.kernels.hip.cpp
disvert.kernels.hip.cpp
euler_scheme.kernels.hip.cpp
hevi_scheme.kernels.hip.cpp
transfert_mpi_legacy.kernels.hip.cpp
dissip_gcm.kernels.hip.cpp
advect_tracer.kernels.hip.cpp
advect.kernels.hip.cpp
""".strip().split("\n")

for f in files:
    content = open(f).read()
    new = kernelLaunch.transformString(content)
    suffix = ""# "TEST"
    with open(f+suffix,"w") as out:
        out.write(new)
