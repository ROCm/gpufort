# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os, sys, traceback
import copy
import re

import orjson

from gpufort import util

from . import opts
from . import indexer
from . import types

def _default_implicit_type(var_expr):
    if var_expr[0] in "ijklmn":
        return "integer", None
    else:
        return "real", None

def _implicit_type(var_expr,implicit_none,type_map):
    """
    :param dict type_map: contains a tuple of Fortran type and kind
                          for certain letters.
    :param bool implicit_none: 
    """
    if var_expr.isidentifier(): 
        if len(type_map) and var_expr[0] in type_map:
            return type_map[var_expr[0]]
        elif var_expr[0:2] == "_i":
            return "integer", None
        elif not implicit_none:
            return _default_implicit_type(var_expr)
        else:
            raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))
    else:
        raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))

def _lookup_implicitly_declared_var(var_expr,implicit_none,type_map={}):
    """
    :param dict type_map: contains a tuple of Fortran type and kind
                          for certain letters.
    :param bool implicit_none: 
    """
    f_type, kind = _implicit_type(var_expr,implicit_none,type_map)
    if kind != None:
        f_type_full = "".join([f_type,"(",kind,")"])
    else:
        f_type_full = f_type
    # TODO len might not be None if a character is specified
    # TODO params might not be [] if a type with parameters is specified
    return types.create_index_var(f_type,None,kind,[],var_expr)

def condense_only_groups(iused_modules):
    """Group consecutive used modules with same name
    if both have non-empty 'only' list.
    """
    result = []
    for iused_module in iused_modules:
        if not len(result):
            result.append(iused_module)
        else:
            #print(iused_module)
            last = result[-1]
            if (iused_module["name"] == last["name"]
               and iused_module["attributes"] == last["attributes"]
               and (len(iused_module["only"])>0) 
                   and (len(last["only"])>0)):
                last["only"] += iused_module["only"] # TODO check for duplicates
            else:
                result.append(iused_module)
    return result

def condense_non_only_groups(iused_modules):
    """Group consecutive used modules with same name
    if both have no 'only' list.

    Background:

    Given a consecutive list of use statements that include the
    same module, one can group them together to just two use statements.

    Example (as seen in WRF):

    ```
    use a, b1 => a1 ! use all of a with orig. name except a1 which shall be named b1
    use a, b2 => a2 ! use all of a with orig. name except a2 which shall be named b2
    use a, b3 => a3 ! use all of a with orig. name except a3 which shall be named b3; 
                    ! now a1,a2 are accessible via orig. name too but not a3

    ```
    can be transformed to

    ```
    use a, only: b1 => a1, b2 => a2
    use a, b3 => a3
    ```
    """

    # - remove `use <mod>` statements before the end of the list
    # - combine renamings of `use <mod>`, statements before the end of the list
    # - for the time being, ignore `use <mod>, only: <only-list>` statements in the interior of the list (no use case)
    # group
    groups = []
    for iused_module in iused_modules:
        if not len(groups):
            groups.append([iused_module])
        else:
            last = groups[-1]
            if (iused_module["name"] == last[0]["name"] 
               and iused_module["attributes"] == last[0]["attributes"]
               and (len(iused_module["only"])==0) 
                   and (len(last[0]["only"])==0)):
                last.append(iused_module)
            else:
                groups.append([iused_module])
    # combine
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            entry1 = copy.deepcopy(group[0])
            entry1["renamings"] = []
            for iused_module in group[:-1]: # exclude last
                 entry1["only"] += iused_module["renamings"]
            entry2 = group[-1]
            result.append(entry1)
            result.append(entry2)
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def _get_accessibility(ientry,iparent):
    """
    """
    if ("private" in ientry["attributes"]
       or ientry["name"] in iparent["private"]):
        return "private"
    elif ("public" in ientry["attributes"]
       or ientry["name"] in iparent["public"]):
        return "public"
    else:
        return iparent["accessibility"]

@util.logging.log_entry_and_exit(opts.log_prefix)
def _resolve_dependencies(scope,
                          index_record,
                          index,
                          strict_checks = False):
    """Include variable, type, and procedure records from modules used
    by the current record (module,program or procedure).

    Per used module ('parent'), we need to recursively lookup other modules ('children') whose definitions are used
    by the parent. Only publicly accessible definitions must be included from the children.
    Like the parent's own definitions, the childrens' definitions are then subject to the parent's
    accessibility rules.

    :param dict scope: the scope that you updated with information from the used modules.
    :param dict index_record: a module/program/procedure index record
    :param list index: list of module/program index records
    """
    def add_to_scope_(scope,scope_additions):
        for entry_type in types.SCOPE_ENTRY_TYPES:
            scope[entry_type] += scope_additions[entry_type]
    
    def handle_use_statements_(icurrent,depth=0):
        """
        recursive function
        :param dict icurrent: 
        """
        nonlocal index
        indent="-"*depth

        util.logging.log_debug2(
            opts.log_prefix,
            "_resolve_dependencies.handle_use_statements",
            "{}process use statements of module '{}'".format(
                indent,
                icurrent["name"]))

        current_scope = copy.deepcopy(types.EMPTY_SCOPE)
        for used_module in condense_non_only_groups(
                             condense_only_groups(
                               icurrent["used_modules"])):
            #print(used_module)
            # include definitions from other modules
            used_module_ignored = ("intrinsic" in used_module["attributes"]
                                  or used_module["name"] in opts.module_ignore_list)
            used_module_found = False 
            if not used_module_ignored:
                iother = next((imod for imod in index if imod["name"]==used_module["name"]),None)
                used_module_found = iother != None
            if used_module_found:
                # depth first search
                other_scope_copy = copy.deepcopy(create_scope(index, iother["name"])) # recursive, deepcopy to not modify cached scopes
                include_all_entries = not len(used_module["only"])
                if include_all_entries: # simple include
                    util.logging.log_debug2(
                        opts.log_prefix,
                        "_resolve_dependencies.handle_use_statements",
                        "{}use all definitions from module '{}'".format(
                            indent,
                            iother["name"]))
                    # TODO check implications of always including in context of implicit attributes
                    # 1. rename particular definitions found in the other scope
                    if len(used_module["renamings"]):
                        for mapping in used_module["renamings"]:
                            number_of_entries_found = 0
                            for entry_type in types.SCOPE_ENTRY_TYPES:
                                for entry in other_scope_copy[entry_type]:
                                    if entry["name"] == mapping["original"]:
                                        entry["name"] = mapping["renamed"]
                                        util.logging.log_debug2(opts.log_prefix,
                                          "_resolve_dependencies.handle_use_statements",
                                          "{}use {} '{}' from module '{}' as '{}'".format(
                                          indent,
                                          entry_type[0:-1],mapping["original"],
                                          iother["name"],
                                          mapping["renamed"]))
                                        number_of_entries_found += 1
                            # emit error if nothing could be found; note that some definitions might
                            # stem from third-party modules. TODO introduce ignore list
                            if strict_checks and number_of_entries_found < 1:
                                raise util.error.LookupError("no public index record found for '{}' in module '{}'".format(mapping["original"],iother["name"]))
                    add_to_scope_(current_scope,other_scope_copy) 
                else:#(include_all_entries) - select only particular entries
                    for mapping in used_module["only"]:
                        number_of_entries_found = 0
                        for entry_type in types.SCOPE_ENTRY_TYPES:
                            for entry in [irecord for irecord in other_scope_copy[entry_type]  # must be the scope
                                          if _get_accessibility(irecord,iother) == "public"]:
                                if entry["name"] == mapping["original"]:
                                    util.logging.log_debug2(opts.log_prefix,
                                      "_resolve_dependencies.handle_use_statements",
                                      "{}only use {} '{}' from module '{}' as '{}'".format(
                                      indent,
                                      entry_type[0:-1],mapping["original"],
                                      iother["name"],
                                      mapping["renamed"]))
                                    entry["name"] = mapping["renamed"]
                                    current_scope[entry_type].append(entry)
                                    number_of_entries_found += 1
                        # emit error if nothing could be found; note that some definitions might
                        # stem from third-party modules. TODO introduce ignore list
                        if strict_checks and number_of_entries_found < 1:
                            raise util.error.LookupError("no public index record found for '{}' in module '{}'".format(mapping["original"],iother["name"]))
            elif not used_module_ignored:
                msg = "{}no index record found for module '{}' and module is not on ignore list".format(
                    indent,
                    used_module["name"])
                raise util.error.LookupError(msg)
        if icurrent["kind"] == "module" and depth > 0:
            # Apply the accessibility of the current module
            filtered_scope = copy.deepcopy(types.EMPTY_SCOPE)
            for entry_type in types.SCOPE_ENTRY_TYPES:
                filtered_scope[entry_type] += [irecord for irecord in current_scope[entry_type] 
                                              if _get_accessibility(irecord,icurrent) == "public"]
            return filtered_scope
        else:
            return current_scope
    add_to_scope_(scope, handle_use_statements_(index_record))


@util.logging.log_entry_and_exit(opts.log_prefix)
def _search_scope_for_type_or_procedure(scope,
                                        entry_name,
                                        entry_type,
                                        empty_record,
                                        ):
    """
    :param str entry_type: either 'types' or 'procedures'
    """
    util.logging.log_enter_function(opts.log_prefix,"_search_scope_for_type_or_procedure",\
      {"entry_name": entry_name,"entry_type": entry_type})

    # reverse access such that entries from the inner-most scope come first
    scope_entities = reversed(scope[entry_type])

    entry_name_lower = entry_name.lower()
    result = next((entry for entry in scope_entities
                   if entry["name"] == entry_name_lower), None)
    if result is None:
        msg = "no index record found for {} '{}'".format(entry_type[:-1], entry_name)
        raise util.error.LookupError(msg)
    else:
        util.logging.log_debug2(opts.log_prefix,"_search_scope_for_type_or_procedure",\
          "index record found for {} '{}'".format(entry_type[:-1],entry_name))
        util.logging.log_leave_function(
            opts.log_prefix, "_search_scope_for_type_or_procedure")
        return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def _search_index_for_type_or_procedure(index, parent_tag, entry_name,
                                                entry_type, empty_record):
    """
    :param str entry_type: either 'types' or 'procedures'
    """
    util.logging.log_enter_function(opts.log_prefix,"_search_index_for_type_or_procedure",\
      {"parent_tag": parent_tag,"entry_name": entry_name,"entry_type": entry_type})

    if parent_tag is None:
        scope = dict(types.EMPTY_SCOPE) # top-level subroutine/function
        scope["procedures"] = [index_entry for index_entry in index\
          if index_entry["name"]==entry_name and index_entry["kind"] in ["subroutine","function"]]
    else:
        scope = create_scope(index, parent_tag)
    return _search_scope_for_type_or_procedure(scope, entry_name,
                                                       entry_type,
                                                       empty_record)


# API
@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index_search_tag_for_var(var_expr):
    return util.parsing.strip_array_indexing(var_expr).lower()

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_scope_from_declaration_list(declaration_list_snippet,
                                       preproc_options=""):
    """
    :param str declaration_section_snippet: A snippet that contains solely variable and type
    declarations.
    """
    dummy_module = """module dummy\n{}
    end module dummy""".format(declaration_list_snippet)
    index = indexer.create_index_from_snippet(dummy_module, preproc_options=preproc_options)
    scope = create_scope(index, "dummy")
    return scope

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index_from_scope(scope):
    """Creates an index that contains a single module
       that has the scope's tag as module name (':' replaced by '_').
."""
    imodule = { "name": scope["tag"].replace(":","_"), "kind": "module" }
    for entry_type in types.SCOPE_ENTRY_TYPES:
        imodule[entry_type] = copy.deepcopy(scope[entry_type])
    return [ imodule ]

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_scope(index, tag):
    """
    :param str tag: a colon-separated list of strings. Ex: mymod:mysubroutine or mymod.
    :note: not thread-safe
    :note: tries to reuse existing scopes.
    :note: assumes that number of scopes will be small per file. Hence, uses list instead of tree data structure
           for storing scopes.
    """

    # check if already a scope exists for the tag or if
    # it can be derived from a higher-level scope
    existing_scope = types.EMPTY_SCOPE
    nesting_level = -1 # -1 implies that nothing has been found
    scopes_to_delete = []
    tag_tokens = tag.split(":")
    for s in opts.scopes:
        existing_tag = s["tag"]
        existing_tag_tokens = existing_tag.split(":")
        len_existing_tag_tokens=len(existing_tag_tokens)
        if tag_tokens[0:len_existing_tag_tokens] == existing_tag_tokens[0:len_existing_tag_tokens]:
            existing_scope = s
            nesting_level = len_existing_tag_tokens - 1
        else:
            scopes_to_delete.append(s)
    # clean up scopes that are not used anymore
    if opts.remove_outdated_scopes and len(scopes_to_delete):
        util.logging.log_debug(opts.log_prefix,"create_scope",\
          "delete outdated scopes with tags '{}'".format(\
            ", ".join([s["tag"] for s in scopes_to_delete])))
        for s in scopes_to_delete:
            opts.scopes.remove(s)

    # return existing existing_scope or create it
    if len(tag_tokens) - 1 == nesting_level:
        util.logging.log_debug(opts.log_prefix,"create_scope",\
          "found existing scope for tag '{}'".format(tag))
        util.logging.log_debug4(opts.log_prefix,"create_scope",\
          "variables in scope: {}".format(", ".join([var["name"] for var in existing_scope["variables"]])))
        return existing_scope
    else:
        new_scope = copy.deepcopy(existing_scope)
        new_scope["tag"] = tag

        # we already have a scope for this record
        if nesting_level >= 0:
            base_record_tag = ":".join(tag_tokens[0:nesting_level + 1])
            util.logging.log_debug(opts.log_prefix,"create_scope",\
              "create scope for tag '{}' based on existing scope with tag '{}'".format(tag,base_record_tag))
            base_record = next((
                module for module in index if module["name"] == tag_tokens[0]),
                               None)
            for l in range(1, nesting_level + 1):
                base_record = next(
                    (procedure for procedure in base_record["procedures"]
                     if procedure["name"] == tag_tokens[l]), None)
            current_record_list = base_record["procedures"]
        else:
            util.logging.log_debug(opts.log_prefix,"create_scope",\
              "create scope for tag '{}'".format(tag))
            current_record_list = index
            # add top-level procedures to scope of top-level entry
            new_scope["procedures"] += [index_entry for index_entry in index\
                    if index_entry["kind"] in ["subroutine","function"] and\
                       index_entry["name"] != tag_tokens[0]]
            util.logging.log_debug(opts.log_prefix,"create_scope",\
              "add {} top-level procedures to scope".format(len(new_scope["procedures"])))
        begin = nesting_level + 1 #

    
        for d in range(begin, len(tag_tokens)):
            searched_name = tag_tokens[d]
            for current_record in current_record_list:
                if current_record["name"] == searched_name:
                    # 1. first include definitions from used records
                    _resolve_dependencies(new_scope, current_record,
                                          index)
                    # 2. now include the current record's definitions
                    for entry_type in types.SCOPE_ENTRY_TYPES:
                        if entry_type in current_record:
                            new_scope[entry_type] += current_record[entry_type]
                    #print("{}:{}".format(":".join(tag_tokens),[p["name"] for p in new_scope["procedures"]]))
                    current_record_list = current_record["procedures"]
                    break
        opts.scopes.append(new_scope)
        util.logging.log_leave_function(opts.log_prefix, "create_scope")
        return new_scope

@util.logging.log_entry_and_exit(opts.log_prefix)
def _lookup_index_record_hierarchy(scope_tag):
    """Given a scope tag `tag1:tag2:...:tagn`, this routine
    will return a list of the n index records beginning from the
    top level parent program/module/function/subroutine.
    """
    scope_tag_tokens = scope_tag.split(":")
    current = None
    result = []
    for i,_ in enumerate(scope_tag_tokens):
        if i == 0:
            current = next((ientry for ientry in self.index if ientry["name"] == scope_tag_tokens[0]),None)
        else:
            parent = current
            current = next((ientry for ientry in parent["procedures"] 
                          if ientry["name"] == scope_tag_tokens[i]),None)
        if current == None:
            raise ValueError("could not find index record for tag '{}'".format(scope_tag_tokens[0:i+1]))
        result.append(current)
    return result

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_var(scope,
                         var_expr):
    """
    %param str variable_tag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c' or 'a%b(i,j)%c(a%i5)'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_scope_for_var",\
      {"var_expr": var_expr})

    #print(scope["tag"])
    #print([v["name"] for v in scope["variables"]])

    result = None
    # reverse access such that entries from the inner-most scope come first
    scope_types = list(reversed(scope["types"]))

    variable_tag = create_index_search_tag_for_var(var_expr)
    list_of_var_names = variable_tag.split("%")
    
    def lookup_from_left_to_right_(scope_vars, pos=0):
        """:note: recursive"""
        nonlocal scope_types
        nonlocal list_of_var_names

        var_name = list_of_var_names[pos]
        if pos == len(list_of_var_names) - 1:
            result = next(
                (var for var in scope_vars if var["name"] == var_name),
                None)
            if result == None:
                raise util.error.LookupError("no index record found for variable tag '{}' in scope".format(variable_tag))
        else:
            matching_type_var = next((
                var for var in scope_vars if var["name"] == var_name),
                                     None)
            if matching_type_var == None:
                raise util.error.LookupError("no index record found for variable tag '{}' in scope".format(variable_tag))
            matching_type = next(
                (typ for typ in scope_types
                 if typ["name"] == matching_type_var["kind"]), None)
            if matching_type == None:
                raise util.error.LookupError("no index record found for derived type '{}' in scope".format(matching_type_var["kind"]))
            result = lookup_from_left_to_right_(
                reversed(matching_type["variables"]), pos + 1)
        return result

    try:
        result = lookup_from_left_to_right_(reversed(scope["variables"]))
    except util.error.LookupError as e:
        #index_record = _lookup_index_record_hierarchy(scope["tag"])[-1]
        #implicit_spec = index_record["implicit"]
        result = _lookup_implicitly_declared_var(var_expr,implicit_none=True,type_map={})

    util.logging.log_debug2(opts.log_prefix,"search_scope_for_var",\
      "entry found for variable tag '{}'".format(variable_tag))
    util.logging.log_leave_function(opts.log_prefix,
                                    "search_scope_for_var")
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_type(scope, type_name):
    """
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    result = _search_scope_for_type_or_procedure(
        scope, type_name, "types", types.EMPTY_TYPE)
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_procedure(scope, procedure_name):
    """
    :param str procedure_name: lower case name of the searched procedure. Simple identifier such as 'mysubroutine'.
    """
    result = _search_scope_for_type_or_procedure(
        scope, procedure_name, "procedures", types.EMPTY_PROCEDURE)
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_index_for_var(index,
                         parent_tag,
                         var_expr):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    %param str var_expr% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c'. Note that all array indexing expressions must be stripped away.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_var",\
      {"parent_tag": parent_tag,"var_expr": var_expr})

    scope = create_scope(index, parent_tag)
    try:
        result = search_scope_for_var(scope, var_expr)
        util.logging.log_leave_function(opts.log_prefix, "search_index_for_var")
        return result
    except util.error.LookupError as e:
        msg = e.args[0]+" (scope tag: '{}')".format(parent_tag)
        e.args = (msg, )
        raise

def search_index_for_type(index, parent_tag, type_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_type",\
      {"parent_tag": parent_tag,"type_name": type_name})
    try:
        result = _search_index_for_type_or_procedure(
            index, parent_tag, type_name, "types", types.EMPTY_TYPE)
        util.logging.log_leave_function(opts.log_prefix, "search_index_for_type")
        return result
    except util.error.LookupError as e:
        msg = e.args[0]+" (scope tag: '{}')".format(parent_tag)
        e.args = (msg, )
        raise

def search_index_for_procedure(index, parent_tag, procedure_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str procedure_name: lower case name of the searched procedure. Simple identifier such as 'mysubroutine'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_procedure",\
      {"parent_tag": parent_tag,"procedure_name": procedure_name})
    try:
        result = _search_index_for_type_or_procedure(
            index, parent_tag, procedure_name, "procedures", types.EMPTY_PROCEDURE)
        util.logging.log_leave_function(opts.log_prefix,
                                        "search_index_for_procedure")
        return result
    except util.error.LookupError as e:
        msg = e.args[0]+" (scope tag: '{}')".format(parent_tag)
        e.args = (msg, )
        raise
