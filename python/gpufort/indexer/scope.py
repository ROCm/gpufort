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

def __default_implicit_type(var_expr):
    if var_expr[0] in "ijklmn":
        return "integer", None
    else:
        return "real", None

def __implicit_type(var_expr,implicit_none,type_map):
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
            return __default_implicit_type(var_expr)
        else:
            raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))
    else:
        raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))

def __lookup_implicitly_declared_var(var_expr,implicit_none,type_map={}):
    """
    :param dict type_map: contains a tuple of Fortran type and kind
                          for certain letters.
    :param bool implicit_none: 
    """
    f_type, kind = __implicit_type(var_expr,implicit_none,type_map)
    return types.create_index_var(f_type,kind,var_expr)

@util.logging.log_entry_and_exit(opts.log_prefix)
def _resolve_dependencies(scope,
                          index_record,
                          index):
    """Include variable, type, and procedure records from modules used
    by the current record (module,program or procedure).

    :param dict scope: the scope that you updated with information from the used modules.
    :param dict index_record: a module/program/procedure index record
    :param list index: list of module/program index records

    TODO must be recursive!!!
    """

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def handle_use_statements_(scope, imodule):
        """
        recursive function
        :param dict imodule: 
        """
        nonlocal index
        for used_module in imodule["used_modules"]:
            used_module_found = used_module["name"] in opts.module_ignore_list
            # include definitions from other modules
            for module in index:
                if module["name"] == used_module["name"]:
                    handle_use_statements_(scope, module) # recursivie call

                    used_module_found = True
                    include_all_entries = not len(used_module["only"])
                    if include_all_entries: # simple include
                        util.logging.log_debug2(
                            opts.log_prefix,
                            "_resolve_dependencies.handle_use_statements",
                            "use all definitions from module '{}'".format(
                                imodule["name"]))
                        for entry_type in types.SCOPE_ENTRY_TYPES:
                            scope[entry_type] += module[entry_type]
                    else:
                        for mapping in used_module["only"]:
                            for entry_type in types.SCOPE_ENTRY_TYPES:
                                for entry in module[entry_type]:
                                    if entry["name"] == mapping["original"]:
                                        util.logging.log_debug2(opts.log_prefix,
                                          "_resolve_dependencies.handle_use_statements",\
                                          "use {} '{}' as '{}' from module '{}'".format(\
                                          entry_type[0:-1],mapping["original"],mapping["renamed"],\
                                          imodule["name"]))
                                        copied_entry = copy.deepcopy(entry)
                                        copied_entry["name"] = mapping[
                                            "renamed"]
                                        scope[entry_type].append(copied_entry)
            if not used_module_found:
                msg = "no index record for module '{}' could be found".format(
                    used_module["name"])
                raise util.error.LookupError(msg)

    handle_use_statements_(scope, index_record)


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
        msg = "no entry found for {} '{}'.".format(entry_type[:-1], entry_name)
        raise util.error.LookupError(msg)
    else:
        util.logging.log_debug2(opts.log_prefix,"_search_scope_for_type_or_procedure",\
          "entry found for {} '{}'".format(entry_type[:-1],entry_name))
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
    for s in opts.scopes:
        existing_tag = s["tag"]
        if existing_tag == tag[0:len(existing_tag)]:
            existing_scope = s
            nesting_level = len(existing_tag.split(":")) - 1
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
    tag_tokens = tag.split(":")
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
                    # 1. first include variables from included
                    _resolve_dependencies(new_scope, current_record,
                                                 index)
                    # 2. now include the current record's
                    for entry_type in types.SCOPE_ENTRY_TYPES:
                        if entry_type in current_record:
                            new_scope[entry_type] += current_record[entry_type]
                    current_record_list = current_record["procedures"]
                    break
        opts.scopes.append(new_scope)
        util.logging.log_leave_function(opts.log_prefix, "create_scope")
        return new_scope


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_scope_for_var(scope,
                         var_expr,
                         resolve=False):
    """
    %param str variable_tag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c' or 'a%b(i,j)%c(a%i5)'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_scope_for_var",\
      {"var_expr": var_expr})

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
                raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))
        else:
            matching_type_var = next((
                var for var in scope_vars if var["name"] == var_name),
                                     None)
            if matching_type_var == None:
                raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_expr))
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
        # TODO check what implicit rules are set, should be set in index and scope
        # if scope["implicit"] == "none"
        # elif scope["implicit"] == "default"
        # elif scope["implicit"] == ... 
        result = __lookup_implicitly_declared_var(var_expr,implicit_none=True,type_map={})
    # resolve
    if resolve:
        pass
    # TODO revisit
    #    for ivar in reversed(scope["variables"]):
    #        if "parameter" in ivar["qualifiers"]:
    #            for entry in [
    #                    "kind", "unspecified_bounds", "lbounds", "counts",
    #                    "total_count", "total_bytes", "index_macro"
    #            ]:
    #                if entry in result:
    #                    dest_tokens = util.parsing.tokenize(result[entry])
    #                    modified_entry = ""
    #                    # TODO handle selected kind here
    #                    for tk in dest_tokens:
    #                        modified_entry += tk.replace(
    #                            ivar["name"], "(" + ivar["value"] + ")")
    #                    result[entry] = modified_entry
    #        if "parameter" in result["qualifiers"]:
    #            if not result["f_type"] in ["character", "type"]:
    #                result["value"].replace(ivar["value"],
    #                                        "(" + ivar["value"] + ")")
    #    for entry in [
    #            "value", "kind", "unspecified_bounds", "lbounds", "counts",
    #            "total_count", "total_bytes", "index_macro"
    #    ]:
    #        if entry in result:
    #            entry_value = result[entry]
    #            try:
    #                code = compile(entry_value, "<string>", "eval")
    #                entry_value = str(eval(code, {"__builtins__": {}}, {}))
    #            except:
    #                pass
    #            result[entry] = entry_value

    util.logging.log_debug2(opts.log_prefix,"search_scope_for_var",\
      "entry found for variable '{}'".format(variable_tag))
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
                         var_expr,
                         resolve=False):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    %param str var_expr% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c'. Note that all array indexing expressions must be stripped away.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_var",\
      {"parent_tag": parent_tag,"var_expr": var_expr})

    scope = create_scope(index, parent_tag)
    return search_scope_for_var(scope, var_expr, resolve)


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_index_for_type(index, parent_tag, type_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_type",\
      {"parent_tag": parent_tag,"type_name": type_name})
    result = _search_index_for_type_or_procedure(
        index, parent_tag, type_name, "types", types.EMPTY_TYPE)
    util.logging.log_leave_function(opts.log_prefix, "search_index_for_type")
    return result


@util.logging.log_entry_and_exit(opts.log_prefix)
def search_index_for_procedure(index, parent_tag, procedure_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str procedure_name: lower case name of the searched procedure. Simple identifier such as 'mysubroutine'.
    """
    util.logging.log_enter_function(opts.log_prefix,"search_index_for_procedure",\
      {"parent_tag": parent_tag,"procedure_name": procedure_name})
    result = _search_index_for_type_or_procedure(
        index, parent_tag, procedure_name, "procedures", types.EMPTY_PROCEDURE)
    util.logging.log_leave_function(opts.log_prefix,
                                    "search_index_for_procedure")
    return result
