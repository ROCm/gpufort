# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import indexer

from .. import analysis
from .. import tree

class MaskedCode:
    """This class associates C code with a mask that enables it.
    """

    def __init__(self,code,mask,indent=""):
        """
        :param str code: C/C++ code.
        :param str mask: condition that enables the code, or None. None is treated as `true`.
        """
        self.mask = mask
        self.code = code
        self.indent = indent
    def mask_matches(self,other):
        if self.mask == other.mask:
            return True
        elif not self.has_mask and not other.has_mask:
            return True
        return False
    @property
    def has_mask(self):
        return self.mask not in [None,"true",""]
    def str(self):
        return textwrap.indent(self.code,self.indent)

class MaskedCodeList:

    def __init__(self):
        self._masked_code_list = []
    
    #self._resource_filter.statement_selection_condition(),
    #self._indent
    def add_masked_code(self,code,mask,indent):
        self._masked_code_list.append(
          MaskedCode(
            code,
            mask,
            indent
          )
        )
    
    def add_unmasked_code(self,code,indent):
        self._masked_code_list.append(
          MaskedCode(code,None,indent)
        )

    def _render_mask_open(self,mask,indent=""):
        return textwrap.indent(
          "GPUFORT_MASK_SET ( {} )\n".format(mask),
          indent
        )
    
    def _render_mask_close(self,mask,indent=""):
        return textwrap.indent(
          "GPUFORT_MASK_UNSET ( {} )\n".format(mask),
          indent
        )
    
    def render(self):
        """:note:Detects contiguous blocks of statements with the same mask."""
        result = "" 
        previous = MaskedCode(None,None)
        for masked_code in self._masked_code_list:   
            if not masked_code.mask_matches(previous): 
                if previous.has_mask:
                   result += self._render_mask_close(previous.mask)
                if masked_code.has_mask:
                   result += self._render_mask_open(masked_code.mask)
            result += masked_code.str().rstrip("\n") + "\n"
            previous = masked_code
        if masked_code.has_mask:
            result += self._render_mask_close(masked_code.mask)
        return result

def render_private_vars_decl_list(ttvalues,scope):
    decl_list_snippet = ""
    for private_var in ttvalues:
        var_expr = tree.traversals.make_fstr(private_var)
        var_tag = indexer.scope.create_index_search_tag(var_expr) 
        if "%" in var_tag:
            # todo: get rid of this limitation,  if necessary
            raise error.LimitationError("private var must not be derived type member")
        tavar = analysis.create_analysis_var(scope,var_expr) 
        c_prefix = "__shared__ " if "shared" in tavar["attributes"] else ""
        c_type = tavar["kind"] if tavar["f_type"]=="type" else tavar["c_type"]
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
