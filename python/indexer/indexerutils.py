# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import indexer.indexer as indexer
import indexer.scoper as scoper
import linemapper.linemapper as linemapper

def update_index_from_snippet(index,snippet,preproc_options=""):
    macro_stack = linemapper.init_macros(preproc_options)
    linemaps    = linemapper.preprocess_and_normalize(snippet.split("\n"),"dummy.f90",macro_stack)
    indexer.update_index_from_linemaps(linemaps,index)

def create_index_from_snippet(snippet,preproc_options):
    index       = []
    update_index_from_snippet(index,snippet,preproc_options="")
    return index

def create_scope_from_declaration_list(declaration_list_snippet,preproc_options=""):
    """
    :param str declaration_section_snippet: A snippet that contains solely variable and type
    declarations.
    """
    dummy_module = """module dummy\n{}
    end module dummy""".format(declaration_list_snippet)
    index = create_index_from_snippet(dummy_module,preproc_options)
    scope = scoper.create_scope(index,"dummy")
    return scope

def search_derived_types(imodule,search_subprograms=True):
    """Search through a module (or program)
    and return derived type definition index entries.
    """
    global LOG_PREFIX

    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_search_derived_types",\
      {"modulename":imodule.get("name",""),\
       "search_subprograms":str(search_subprograms)})
    
    itypes = {}
    prefix = ""   
    def collect_types_(irecord):
        nonlocal itypes
        nonlocal prefix
        prefix += irecord["name"]
        for itype in irecord["types"]:
            ident = prefix + "_" + itype["name"]
            itypes[ident] = itype
        if search_subprograms:
            for isubprogram in irecord["subprograms"]:
                collect_types_(isubprogram)
    collect_types_(imodule)

    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_search_derived_types",
      {"typenames":str([itype["name"] for itype in itypes.values()])})
    
    return itypes