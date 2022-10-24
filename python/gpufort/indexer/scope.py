# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os, sys, traceback
import copy
import re

from gpufort import util

from . import opts
from . import indexer
from . import indexertypes

def _lookup_implicitly_declared_var(var_expr,implicit_specs):
    """
    :param dict type_map: contains a tuple of Fortran type and kind
                          for certain letters.
    :param bool implicit_none: 
    """
    var_expr_lower = var_expr.lower()
    if var_expr_lower.isidentifier(): 
        # todo: support arrays
        if var_expr_lower[0:2] == "_i":
            f_type,f_len,kind = "integer", None, None
            return indexertypes.create_index_var(f_type,
                                          f_len,
                                          kind,
                                          [],
                                          var_expr_lower)
        for spec in implicit_specs:
            if var_expr_lower[0] in spec["letters"]:
                return indexertypes.create_index_var(spec["f_type"],
                                              spec["len"],
                                              spec["kind"],
                                              [],
                                              var_expr_lower)
        raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))
    else:
        raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))

def combine_use_statements(iused_modules):
    """Group used modules with same name.
    If all modules in a group have an 'only' list, then
    the 'only' lists of all modules are combined.
    In any other case, the renaming lists and only
    lists are combined to a single renaming list.

    :note: Modules with same name and attributes  
    can be combined no matter their position
    in the use statement list as duplicate names
    are not allowed in the same scope by the Fortran standard.
    """
    result = []
    def lookup_existing_(iused_module):
        nonlocal result
        for i,entry in enumerate(result):
            same_name       = entry["name"] == iused_module["name"]
            same_attributes = len(entry["attributes"]) == len(iused_module["attributes"])
            if same_name and same_attributes:
                for j,attrib in entry["attributes"]:
                    if attrib not in iused_module["attributes"]:
                        same_attributes = False
                        break
            if same_name and same_attributes:
                return entry
        return None

    for iused_module in iused_modules:
        existing_entry = lookup_existing_(iused_module)
        if existing_entry == None:
            result.append(copy.deepcopy(iused_module))
        else:
            if len(existing_entry["only"]) and len(iused_module["only"]):
                existing_entry["only"] += copy.deepcopy(iused_module["only"])
            else:
                # combine the lists
                existing_entry["renamings"] += existing_entry["only"] 
                existing_entry["only"].clear()
                existing_entry["renamings"] += (copy.deepcopy(iused_module["only"])
                                               + copy.deepcopy(iused_module["renamings"]))
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
                          strict_checks = False): # make option
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
        for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
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

        current_scope = indexertypes.new_scope()
        for used_module in combine_use_statements(icurrent["used_modules"]):
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
                other_scope_copy = indexertypes.copy_scope(create_scope(index, iother["name"])) # recursive, deepcopy to not modify cached scopes
                include_all_entries = not len(used_module["only"])
                if include_all_entries: # simple include
                    util.logging.log_debug2(
                        opts.log_prefix,
                        "_resolve_dependencies.handle_use_statements",
                        "{}use all definitions from module '{}'".format(
                            indent,
                            iother["name"]))
                    # todo: check implications of always including in context of implicit attributes
                    # 1. rename particular definitions found in the other scope
                    other_variables_to_rename = set(mapping["original"] for mapping in used_module["renamings"])
                    for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
                        for entry in other_scope_copy[entry_type]:  # must be the scope
                            if _get_accessibility(entry,iother) == "public":
                                if entry["name"] in other_variables_to_rename:
                                    for mapping in used_module["renamings"]:
                                        orig_name = mapping["original"]
                                        if entry["name"] == orig_name:
                                            util.logging.log_debug2(opts.log_prefix,
                                              "_resolve_dependencies.handle_use_statements",
                                              "{}use {} '{}' from module '{}' as '{}'".format(
                                              indent,
                                              entry_type[0:-1],orig_name,
                                              iother["name"],
                                              mapping["renamed"]))
                                            if orig_name in other_variables_to_rename: # name might exist multipe times, hiding
                                                other_variables_to_rename.remove(orig_name)
                                            entry["name"] = mapping["renamed"]
                                # always append entries
                                current_scope[entry_type].append(entry)

                    # emit error if nothing could be found; note that some definitions might
                    # stem from third-party modules. TODO introduce ignore list
                    if strict_checks and len(other_variables_to_rename) > 0:
                        raise util.error.LookupError("no public index record found for '{}' in module '{}'".format("','".join(other_variables_to_rename),iother["name"]))
#                    add_to_scope_(current_scope,other_scope_copy,iother) 
                else:#(include_all_entries) - select only particular entries
                    other_variables_to_use = set(mapping["original"] for mapping in used_module["only"])
                    for mapping in used_module["only"]:
                        orig_name = mapping["original"]
                        for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
                            for entry in other_scope_copy[entry_type]:
                                if _get_accessibility(entry,iother) == "public":
                                    if entry["name"] == orig_name:
                                        util.logging.log_debug2(opts.log_prefix,
                                          "_resolve_dependencies.handle_use_statements",
                                          "{}only use {} '{}' from module '{}' as '{}'".format(
                                          indent,
                                          entry_type[0:-1],orig_name,
                                          iother["name"],
                                          mapping["renamed"]))
                                        if orig_name in other_variables_to_use: # name might exist multiple times, hiding
                                            other_variables_to_use.remove(orig_name)
                                        entry["name"] = mapping["renamed"]
                                        current_scope[entry_type].append(entry) # only append if in list
                    # emit error if nothing could be found; note that some definitions might
                    # stem from third-party modules. TODO introduce ignore list
                    if strict_checks and len(other_variables_to_use) > 0:
                       raise util.error.LookupError("no public index record found for '{}' in module '{}'".format("','".join(other_variables_to_use),iother["name"]))
            elif not used_module_ignored:
                msg = "{}no index record found for module '{}' and module is not on ignore list".format(
                    indent,
                    used_module["name"])
                raise util.error.LookupError(msg)
        if icurrent["kind"] == "module" and depth > 0:
            # Apply the accessibility of the current module
            filtered_scope = indexertypes.new_scope()
            for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
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
        scope = dict(indexertypes.EMPTY_SCOPE) # top-level subroutine/function
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
    for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
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
    existing_scope = indexertypes.new_scope()
    nesting_level  = -1 # -1 implies that nothing has been found
    tag_tokens = tag.split(":")
    for s in opts.scopes:
        existing_tag = s["tag"]
        existing_tag_tokens = existing_tag.split(":")
        len_existing_tag_tokens=len(existing_tag_tokens)
        if tag_tokens[0:len_existing_tag_tokens] == existing_tag_tokens[0:len_existing_tag_tokens]:
            existing_scope = s
            nesting_level = len_existing_tag_tokens - 1

    # return existing existing_scope or create it
    if len(tag_tokens) - 1 == nesting_level:
        util.logging.log_debug(opts.log_prefix,"create_scope",\
          "found existing scope for tag '{}'".format(tag))
        util.logging.log_debug4(opts.log_prefix,"create_scope",\
          "variables in scope: {}".format(", ".join([var["name"] for var in existing_scope["variables"]])))
        return existing_scope
    else:
        new_scope = indexertypes.copy_scope(existing_scope,index,tag)

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
                    scope_additions = indexertypes.new_scope()
                    # 1. first include definitions from used records
                    # todo: ambiguous definitions still possible
                    _resolve_dependencies(scope_additions, current_record,
                                          index)
                    # todo: ambiguous definitions can be detected here
                    # 2. now include the current record's definitions
                    for entry_type in indexertypes.SCOPE_ENTRY_TYPES:
                        if entry_type in current_record:
                            # scope entry has the parent tag as well
                            for index_entry in current_record[entry_type]:
                                scope_entry = copy.deepcopy(index_entry)
                                scope_entry["parent_tag"] = ":".join(tag_tokens[0:d+1])
                                scope_additions[entry_type].append(scope_entry)
                        # remove hidden parent entries
                        # todo: check if variable can hide type / procedure etc
                        for local_entry in scope_additions[entry_type]:
                            for parent_entry in copy.copy(new_scope[entry_type]):
                                if parent_entry["name"] == local_entry["name"]:
                                    new_scope[entry_type].remove(parent_entry)
                        new_scope[entry_type] += scope_additions[entry_type]

                    #print("{}:{}".format(":".join(tag_tokens),[p["name"] for p in new_scope["procedures"]]))
                    current_record_list = current_record["procedures"]
                    break
        # (shallow) copy implicit spec from scope-associated index record
        new_scope["implicit"] = current_record["implicit"]
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
                         var_expr,
                         consider_implicit=True):
    """
    %param str variable_tag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c' or 'a%b(i,j)%c(a%i5)'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_scope_for_var",\
      {"var_expr": var_expr})

    #print(scope["tag"])
    #print([v["name"] for v in scope["variables"]])

    result = None
    # reverse access such that entries from the inner-most scope come first
    #scope_types = list(reversed(scope["types"]))
    scope_types = reversed(scope["types"])

    variable_tag = create_index_search_tag_for_var(var_expr)
    list_of_var_names = variable_tag.split("%")
    
    def lookup_from_left_to_right_(scope_vars, pos=0):
        """:note: recursive"""
        nonlocal scope_types
        nonlocal list_of_var_names

        var_name = list_of_var_names[pos]
        if pos == len(list_of_var_names) - 1: # innermost 
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
            if matching_type["parent_tag"] != scope["tag"]:
                module_scope = create_scope(scope["index"],matching_type["parent_tag"])
                scope_types = reversed(module_scope["types"])
            result = lookup_from_left_to_right_(
                reversed(matching_type["variables"]), pos + 1)
        return result

    try:
        result = lookup_from_left_to_right_(reversed(scope["variables"]))
    except util.error.LookupError as e:
        if not consider_implicit:
            raise e
        else:
            try:
                result = _lookup_implicitly_declared_var(var_expr,scope["implicit"])
            except util.error.LookupError:
                raise e

    util.logging.log_debug2(opts.log_prefix,"search_scope_for_var",\
      "entry found for variable tag '{}'".format(variable_tag))
    util.logging.log_leave_function(opts.log_prefix,
                                    "search_scope_for_var")
    return result

def search_scope_for_implicitly_declared_var(scope,var_expr):
    return _lookup_implicitly_declared_var(var_expr,scope["implicit"])
    

def is_intrinsic(name):
    return name.lower() in intrinsics.intrinsics

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_type(scope, type_name):
    """
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    result = _search_scope_for_type_or_procedure(
        scope, type_name, "types", indexertypes.EMPTY_TYPE)
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_procedure(scope, procedure_name):
    """
    :param str procedure_name: lower case name of the searched procedure. Simple identifier such as 'mysubroutine'.
    """
    result = _search_scope_for_type_or_procedure(
        scope, procedure_name, "procedures", indexertypes.EMPTY_PROCEDURE)
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

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_index_for_type(index, parent_tag, type_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_type",\
      {"parent_tag": parent_tag,"type_name": type_name})
    try:
        result = _search_index_for_type_or_procedure(
            index, parent_tag, type_name, "types", indexertypes.EMPTY_TYPE)
        util.logging.log_leave_function(opts.log_prefix, "search_index_for_type")
        return result
    except util.error.LookupError as e:
        msg = e.args[0]+" (scope tag: '{}')".format(parent_tag)
        e.args = (msg, )
        raise

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_index_for_procedure(index, parent_tag, procedure_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str procedure_name: lower case name of the searched procedure. Simple identifier such as 'mysubroutine'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_procedure",\
      {"parent_tag": parent_tag,"procedure_name": procedure_name})
    try:
        result = _search_index_for_type_or_procedure(
            index, parent_tag, procedure_name, "procedures", indexertypes.EMPTY_PROCEDURE)
        util.logging.log_leave_function(opts.log_prefix,
                                        "search_index_for_procedure")
        return result
    except util.error.LookupError as e:
        msg = e.args[0]+" (scope tag: '{}')".format(parent_tag)
        e.args = (msg, )
        raise

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_value_expr(scope,expr):
    """Search scope for a lrvalue or rvalue expression (str).
    Lookup order: Explicitly-typed variables -> Functions -> Intrinsics -> Implicitly-typed variables
    :param scope: Scope dict.
    :param str expr: An lvalue or rvalue expression such as `a`, `a(i,j)`, `a(i)%b(:)`.  
    :return: Tuple of value type (see indexer.indexertypes.ValueType) 
             and index record if available. (No index record might be
             returned if expression refers to intrinsic)
    """
    value_type = indexertypes.ValueType.UNKNOWN
    index_record = None
    try:
       index_record = search_scope_for_var(scope, expr, 
          consider_implicit = False)
       value_type = indexertypes.ValueType.VARIABLE
    except util.error.LookupError:
        try:
            # todo: check EXTERNAL procedures too 
            index_record = search_scope_for_procedure(scope, expr) # just check if the procedure exists
            value_type = indexertypes.ValueType.PROCEDURE
        except util.error.LookupError:
            if is_intrinsic(expr):
                value_type = indexertypes.ValueType.INTRINSIC
            else:
                try:
                    index_record = _lookup_implicitly_declared_var(var_expr,scope["implicit"])
                    value_type = indexertypes.ValueType.VARIABLE
                except:
                    raise util.error.LookupError("expression '"+expr+"' could not be associated with any variable (explicitly or implicitly declared), procedure, or intrinsic")
    return (value_type, index_record)
