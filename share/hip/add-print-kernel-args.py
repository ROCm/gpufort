# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
from pyparsing import *

# parser grammar
ParserElement.setDefaultWhitespaceChars(" \t\n\r")

identifier = pyparsing_common.identifier

LPAR,RPAR,COMMA,SEMICOLON = map(Suppress, "(),;")
VOID,LAUNCH,AUTO,BBBRA,KKKET = map(Suppress,["void","launch","auto","<<<",">>>"])

arg      = Regex(r"(&|\*)*\w+")
args     = delimitedList(arg)
kernelName = identifier.copy()
kernelLaunch = White() + kernelName + BBBRA + Group(args) + KKKET + LPAR + Group(args) + RPAR + SEMICOLON

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

content="""
  // launch kernel
  krnl_1fb5dc_165<<<*grid,*block,sharedMem,stream>>>(
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
    indent       = tk[0].strip("\n")
    kernelName   = tk[1]
    launchParams = tk[2]
    launchArgs   = tk[3]
    grid         = launchParams[0]
    block        = launchParams[1]
    gridStr  = ",".join(["({0}).{1}".format(grid,ix) for ix in "xyz"])
    blockStr = ",".join(["({0}).{1}".format(block,ix) for ix in "xyz"])
    result  = "\n{0}#if defined(PRINT_KERNEL_ARGS_ALL) || defined(PRINT_KERNEL_ARGS_{1})".format(indent,kernelName)
    result += "\n{0}PRINT_ARGS(\"{1}\",{2},{3},{4},{5});".format(indent,kernelName,gridStr,blockStr,\
            ",".join(launchParams[2:]),",".join(launchArgs))
    result += "\n{0}#endif".format(indent)
    result += "\n{0}{1}<<<{2}>>>({3});".format(indent,kernelName,\
            ",".join(launchParams),\
            ",\n{0}".format(indent+" "*4).join(launchArgs))
    return result

kernelLaunch.setParseAction(parseAction)
#print(args.searchString(content))
#print(kernelLaunch.transformString(content))

for f in files:
    content = open(f).read()
    new = kernelLaunch.transformString(content)
    suffix = ""
    with open(f+suffix,"w") as out:
        out.write(new)