# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import copy

from gpufort import translator

__UNKNOWN = "UNKNOWN"

EMPTY_TYPE = {
    "name": __UNKNOWN, 
    "variables": [],
    "file" : __UNKNOWN,
    "lineno" : -1,
}

EMPTY_PROCEDURE = {
    "kind": __UNKNOWN,
    "name": __UNKNOWN,
    "result_name": __UNKNOWN,
    "attributes": [],
    "dummy_args": [],
    "variables": [],
    "procedures": [],
    "used_modules": [],
    "file" : __UNKNOWN,
    "lineno" : -1,
}

EMPTY_SCOPE = {"tag": "", "types": [], "variables": [], "procedures": []}
SCOPE_ENTRY_TYPES = ["types", "variables", "procedures"]

EMPTY_VAR = {
        "name"   : __UNKNOWN,
        "f_type" : __UNKNOWN,
        "len"    : __UNKNOWN,
        "kind"   : __UNKNOWN,
        "params" : [],
        # TODO bytes per element can be computed on the fly
        "bytes_per_element" : __UNKNOWN,
        "c_type" : __UNKNOWN,
        "attributes" : [],
        # ACC/OMP
        "declare_on_target" : False,
        # arrays
        "bounds" : [],
        "rank"   : -1,
        # parse rhs if necessary
        "rhs" : __UNKNOWN,
        # meta information
        "file" : __UNKNOWN,
        "lineno" : -1,
}

def create_index_var(f_type,f_len,kind,params,name,qualifiers=[],bounds=[],rhs=None,filepath="<unknown>",lineno=-1):
    ivar = copy.deepcopy(EMPTY_VAR)
    # basic
    ivar["name"]        = name
    ivar["f_type"]      = f_type
    ivar["kind"]        = kind
    ivar["len"]         = f_len
    ivar["params"]      = params
    # TODO bytes per element can be computed on the fly
    ivar["attributes"] += qualifiers
    # arrays
    ivar["bounds"] += bounds
    ivar["rank"]   = len(bounds)
    # handle parameters
    #ivar["value"] = None # TODO parse rhs if necessary
    ivar["rhs"] = rhs
    # meta information
    ivar["file"] = filepath
    ivar["lineno"] = lineno
    return ivar

def render_datatype(ivar):
    datatype = ivar["f_type"]
    args = []
    if datatype == "character":
        if ivar["len"] != None:
            args.append(ivar["len"])
    if datatype != "type" and ivar["kind"] != None:
        args.append(ivar["kind"])
    elif datatype == "type":
        arg1 = ivar["kind"]
        if len(ivar["params"]):
            arg1.append("(")
            arg1.append(",".join(ivar["params"]))
            arg1.append(")")
        args.append("".join(arg1))
    if len(args):
        return "".join([datatype,"(",",".join(args),")"])
    else:
        return datatype
    
def render_declaration(ivar):
    result = [render_datatype(ivar)]
    if len(ivar["attributes"]):
        result += [", ",", ".join(ivar["attributes"])]
    result += [" :: ",ivar["name"]]
    if len(ivar["bounds"]):
        result += ["(",", ".join(ivar["bounds"]),")"]
    if ivar["rhs"] != None:
        if "pointer" in ivar["attributes"]:
            result += [" => ",ivar["rhs"]]
        else:
            result += [" = ",ivar["rhs"]]
    return "".join(result)
