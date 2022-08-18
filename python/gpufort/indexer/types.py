# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import copy

EMPTY_TYPE = {
    "name": None, 
    "variables": [],
    "file" : None,
    "lineno" : -1,
}

EMPTY_PROCEDURE = {
    "kind": None,
    "name": None,
    "result_name": None,
    "attributes": [],
    "dummy_args": [],
    "variables": [],
    "procedures": [],
    "used_modules": [],
    "file" : None,
    "lineno" : -1,
}

EMPTY_SCOPE = {"tag": "", "types": [], "variables": [], "procedures": []}
SCOPE_ENTRY_TYPES = ["types", "variables", "procedures"]

EMPTY_VAR = {
        "name"   : None,
        "f_type" : None,
        "len"    : None,
        "kind"   : None,
        "params" : [],
        # TODO bytes per element can be computed on the fly
        "bytes_per_element" : None,
        "c_type" : None,
        "attributes" : [],
        # ACC/OMP
        "declare_on_target" : None,
        # arrays
        "bounds" : [],
        "rank"   : -1,
        # parse rhs if necessary
        "rhs" : None,
        # meta information
        "module": None,
        "file" : None,
        "lineno" : -1,
}

def create_index_var(f_type,
                     f_len,
                     kind,
                     params,
                     name,
                     attributes=[],
                     bounds=[],
                     rhs=None,
                     module=None,
                     filepath="<unknown>",
                     lineno=-1):
    ivar = copy.deepcopy(EMPTY_VAR)
    # basic
    ivar["name"]        = name
    ivar["f_type"]      = f_type
    ivar["kind"]        = kind
    ivar["len"]         = f_len
    ivar["params"]      = params
    # TODO bytes per element can be computed on the fly
    ivar["attributes"] += attributes
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