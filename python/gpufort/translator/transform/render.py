# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import indexer

from .. import analysis

def render_private_variables_decl_list(ttvalues,scope):
    decl_list_snippet = ""
    for private_var in ttvalues
        var_expr = traversals.make_fstr(private_var)
        var_tag = indexer.scope.create_index_search_tag_for_var(var_expr) 
        if "%" in var_tag:
            # TODO get rid of this limitation,  if necessary
            raise error.LimitationError("private var must not be derived type member")
        tavar = analysis.create_analysis_var(scope, 
          indexer.scope.search_scope_for_var(scope,var_targ)
        )
        c_prefix = "__shared__ " if "shared" in tavar.attributes else ""
        c_type = tavar.kind if tavar.f_type=="type" else tavar.c_type
        if tavar["rank"] == 0:
            decl_list_snippet += "{c_prefix}{c_type} {c_name};\n".format(
              c_prefix=c_prefix,
              c_type=c_type,
              c_name = tavar["c_name"]
            )
        else:
            decl_list_snippet += "{c_prefix}{c_type} _{c_name}[{total_size}];\n".format(
              c_prefix=c_prefix,
              c_type=c_type,
              total_size = "*".join(tavar["size"])
            )
            decl_list_snippet += """\
{c_name}.wrap(nullptr,&_{c_name}[0],
  {{tavar["size"] | join(",")}},
  {{tavar["lbounds"] | join(",")}});""".format(
              c_name = tavar
            )
            # array_descr is passed as argument
    return decl_list_snippet

def render_loopnest(loopnest,
                    private_vars = [], # type: list[TTValue]
                    reductions = {}, # type dict[str][list[TTValue]]
                    num_collapse=0, # type: int
                    tile_sizes=[]): # type: list[Union[TTNode,str,int]]
    assert num_collapse <= 1 or len(tile_sizes) == 0, "cannot be specified both"
    if len(loopnest) == num_collapse:
        loopnest_open,\
        loopnest_close,\
        loopnest_resource_filter,\
        loopnest_indent = \
            loopnest.collapse().map_to_hip_cpp()
    elif len(loopnest) == len(tile_sizes):
        loopnest_open,\
        loopnest_close,\
        loopnest_resource_filter,\
        loopnest_indent = \
            loopnest.tile(tile_sizes).map_to_hip_cpp()
    else:
        loopnest_open,\
        loopnest_close,\
        loopnest_resource_filter,\
        loopnest_indent = \
            loopnest.map_to_hip_cpp()
    if len(private_vars):
        loopnest_open += textwrap.indent(
          render_private_variables_decl_list(private_vars,
          loopnest_indent
        )
    # TODO render reduction variables
    # 1. create unique index var with value = loopnest.index()
    # 2. 
    return (loopnest_open,
            loopnest_close,
            loopnest_resource_filter,
            loopnest_indent)
