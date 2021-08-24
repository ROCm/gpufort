# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import copy

import orjson

import utils.logging

# configurable parameters
indexer_dir = os.path.dirname(__file__)
exec(open("{0}/scoper_options.py.in".format(indexer_dir)).read())

ERR_SCOPER_RESOLVE_DEPENDENCIES_FAILED = 1001
ERR_SCOPER_LOOKUP_FAILED = 1002

__UNKNOWN = "UNKNOWN"

EMPTY_VARIABLE = {                         
  "name"                          : __UNKNOWN,
  "f_type"                        : __UNKNOWN,
  "kind"                          : __UNKNOWN,
  "bytes_per_element"               : __UNKNOWN,
  "c_type"                        : __UNKNOWN,
  "f_interface_type"                : __UNKNOWN,
  "f_interface_qualifiers"          : __UNKNOWN,
  "qualifiers"                    : [],
  "declare_on_target"               : __UNKNOWN,
  "rank"                          : -1,
  "unspecified_bounds"             : __UNKNOWN,
  "lbounds"                       : __UNKNOWN,
  "counts"                        : __UNKNOWN,
  "total_count"                    : __UNKNOWN,
  "total_bytes"                    : __UNKNOWN,
  "index_macro"                   : __UNKNOWN,
  "index_macro_with_placeholders" : __UNKNOWN
}

EMPTY_TYPE = {                         
  "name"      : __UNKNOWN,
  "variables" : []
}

EMPTY_SUBPROGRAM = {                         
  "kind"        : __UNKNOWN,
  "name"        : __UNKNOWN,
  "result_name" : __UNKNOWN,
  "attributes"  : [],
  "dummy_args"  : [],
  "variables"   : [],
  "subprograms" : [],
  "used_modules" : []
}

EMPTY_SCOPE = { "tag": "", "types" : [], "variables" : [], "subprograms" : [] } 

__SCOPE_ENTRY_TYPES = ["subprograms","variables","types"]

def _intrnl_resolveDependencies(scope,index_record,index):
    """
    Include variable, type, and subprogram records from modules used
    by the current record (module,program or subprogram).

    :param dict scope: the scope that you updated with information from the used modules.
    :param dict index_record: a module/program/subprogram index record
    :param list index: list of module/program index records

    TODO must be recursive!!!
    """
    global LOG_PREFIX    
    global ERROR_HANDLING

    utils.logging.log_enter_function(LOG_PREFIX,"__resolveDependencies")

    def handle_use_statements_(scope,imodule):
        """
        recursive function
        :param dict imodule: 
        """ 
        nonlocal index
        for used_module in imodule["used_modules"]:
            used_module_found = used_module["name"] in MODULE_IGNORE_LIST
            # include definitions from other modules
            for module in index:
                if module["name"] == used_module["name"]:
                    handle_use_statements_(scope,module) # recursivie call

                    used_module_found   = True
                    include_all_entries = not len(used_module["only"])
                    if include_all_entries: # simple include
                        utils.logging.log_debug2(LOG_PREFIX,"__resolveDependencies.handle_use_statements",
                          "use all definitions from module '{}'".format(imodule["name"]))
                        for entry_type in __SCOPE_ENTRY_TYPES:
                            scope[entry_type] += module[entry_type]
                    else:
                        for mapping in used_module["only"]:
                            for entry_type in __SCOPE_ENTRY_TYPES:
                                for entry in module[entry_type]:
                                    if entry["name"] == mapping["original"]:
                                        utils.logging.log_debug2(LOG_PREFIX,
                                          "__resolveDependencies.handle_use_statements",\
                                          "use {} '{}' as '{}' from module '{}'".format(\
                                          entry_type[0:-1],mapping["original"],mapping["renamed"],\
                                          imodule["name"]))
                                        copied_entry = copy.deepcopy(entry)
                                        copied_entry["name"] = mapping["renamed"]
                                        scope[entry_type].append(copied_entry)
            if not used_module_found:
                msg = "no index record for module '{}' could be found".format(used_module["name"])
                if ERROR_HANDLING == "strict":
                    utils.logging.log_error(LOG_PREFIX,"__resolveDependencies",msg) 
                    sys.exit(ERR_INDEXER_RESOLVE_DEPENDENCIES_FAILED)
                else:
                    utils.logging.log_warning(LOG_PREFIX,"__resolveDependencies",msg)

    handle_use_statements_(scope,index_record)
    utils.logging.log_leave_function(LOG_PREFIX,"__resolveDependencies")

def _intrnl_search_scope_for_typeOrSubprogram(scope,entry_name,entry_type,empty_record):
    """
    :param str entry_type: either 'types' or 'subprograms'
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"__search_scope_for_typeOrSubprogram",\
      {"entry_name":entry_name,"entry_type":entry_type})

    # reverse access such that entries from the inner-most scope come first
    scope_entities = reversed(scope[entry_type])

    entry_nameLower = entry_name.lower()
    result = next((entry for entry in scope_entities if entry["name"] == entry_nameLower),None)  
    if result is None:
        msg = "no entry found for {} '{}'.".format(entry_type[:-1],entry_name)
        if ERROR_HANDLING  == "strict":
            utils.logging.log_error(LOG_PREFIX,"__search_scope_for_typeOrSubprogram",msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logging.log_warning(LOG_PREFIX,"__search_scope_for_typeOrSubprogram",msg) 
        return empty_record, False
    else:
        utils.logging.log_debug2(LOG_PREFIX,"__search_scope_for_typeOrSubprogram",\
          "entry found for {} '{}'".format(entry_type[:-1],entry_name)) 
        utils.logging.log_leave_function(LOG_PREFIX,"__search_scope_for_typeOrSubprogram")
        return result, True

def _intrnl_search_index_for_typeOrSubprogram(index,parent_tag,entry_name,entry_type,empty_record):
    """
    :param str entry_type: either 'types' or 'subprograms'
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"__search_index_for_typeOrSubprogram",\
      {"parent_tag":parent_tag,"entry_name":entry_name,"entry_type":entry_type})

    scope = create_scope(index,parent_tag)
    return _intrnl_search_scope_for_typeOrSubprogram(scope,entry_name,entry_type,empty_record)

# API
def create_scope(index,tag):
    """
    :param str tag: a colon-separated list of strings. Ex: mymod:mysubroutine or mymod.
    :note: not thread-safe
    :note: tries to reuse existing scopes.
    :note: assumes that number of scopes will be small per file. Hence, uses list instead of tree data structure
           for storing scopes.
    """
    global SCOPES
    global REMOVE_OUTDATED_SCOPES
    global MODULE_IGNORE_LIST
    global LOG_PREFIX    
    utils.logging.log_enter_function(LOG_PREFIX,"create_scope",{"tag":tag,"ERROR_HANDLING":ERROR_HANDLING})
    
    # check if already a scope exists for the tag or if
    # it can be derived from a higher-level scope
    existing_scope   = EMPTY_SCOPE
    nesting_level    = -1 # -1 implies that nothing has been found
    scopes_to_delete  = []
    for s in SCOPES:
        existing_tag = s["tag"]
        if existing_tag == tag[0:len(existing_tag)]:
            existing_scope = s
            nesting_level  = len(existing_tag.split(":"))-1
        else:
            scopes_to_delete.append(s)
    # clean up scopes that are not used anymore 
    if REMOVE_OUTDATED_SCOPES and len(scopes_to_delete):
        utils.logging.log_debug(LOG_PREFIX,"create_scope",\
          "delete outdated scopes with tags '{}'".format(\
            ", ".join([s["tag"] for s in scopes_to_delete])))
        for s in scopes_to_delete:
            SCOPES.remove(s)

    # return existing existing_scope or create it
    tag_tokens = tag.split(":")
    if len(tag_tokens)-1 == nesting_level:
        utils.logging.log_debug(LOG_PREFIX,"create_scope",\
          "found existing scope for tag '{}'".format(tag))
        utils.logging.log_leave_function(LOG_PREFIX,"create_scope")
        return existing_scope
    else:
        new_scope = copy.deepcopy(existing_scope)
        new_scope["tag"] = tag 
 
        # we already have a scope for this record
        if nesting_level >= 0:
            base_recordTag = ":".join(tag_tokens[0:nesting_level+1])
            utils.logging.log_debug(LOG_PREFIX,"create_scope",\
              "create scope for tag '{}' based on existing scope with tag '{}'".format(tag,base_recordTag))
            base_record = next((module for module in index if module["name"] == tag_tokens[0]),None)  
            for l in range(1,nesting_level+1):
                base_record = next((subprogram for subprogram in base_record["subprograms"] if subprogram["name"] == tag_tokens[l]),None)
            current_recordList = base_record["subprograms"]
        else:
            utils.logging.log_debug(LOG_PREFIX,"create_scope",\
              "create scope for tag '{}'".format(tag))
            current_recordList = index
            # add top-level subprograms to scope of top-level entry
            new_scope["subprograms"] += [index_entry for index_entry in index\
                    if index_entry["kind"] in ["subroutine","function"] and\
                       index_entry["name"] != tag_tokens[0]]
            utils.logging.log_debug(LOG_PREFIX,"create_scope",\
              "add {} top-level subprograms to scope".format(len(new_scope["subprograms"])))
        begin = nesting_level + 1 # 
        
        for d in range(begin,len(tag_tokens)):
            searched_name = tag_tokens[d]
            for current_record in current_recordList:
                if current_record["name"] == searched_name:
                    # 1. first include variables from included
                    _intrnl_resolveDependencies(new_scope,current_record,index) 
                    # 2. now include the current record's   
                    for entry_type in __SCOPE_ENTRY_TYPES:
                        if entry_type in current_record:
                            new_scope[entry_type] += current_record[entry_type]
                    current_recordList = current_record["subprograms"]
                    break
        SCOPES.append(new_scope)
        utils.logging.log_leave_function(LOG_PREFIX,"create_scope")
        return new_scope

def search_scope_for_variable(scope,variable_tag):
    """
    %param str variable_tag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c'. Note that all array indexing expressions must be stripped away.
    :see: translator.translator.create_index_search_tag_for_variable(variable_expression)
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"search_index_for_variable",\
      {"variable_tag":variable_tag})

    result = None
    # reverse access such that entries from the inner-most scope come first
    scope_types = reversed(scope["types"])

    list_of_var_names = variable_tag.lower().split("%") 
    def lookup_from_left_to_right_(scope_variables,pos=0):
        """
        :note: recursive
        """
        nonlocal scope_types
        nonlocal list_of_var_names
     
        var_name = list_of_var_names[pos]
        if pos == len(list_of_var_names)-1:
            result = next((var for var in scope_variables if var["name"] == var_name),None)  
        else:
            try:
                matching_typeVar = next((var for var in scope_variables if var["name"] == var_name),None)
                matching_type    = next((typ for typ in scope_types if typ["name"] == matching_typeVar["kind"]),None)
                result = lookup_from_left_to_right_(reversed(matching_type["variables"]),pos+1)
            except Exception as e:
                raise e
                result = None
        return result
    result = lookup_from_left_to_right_(reversed(scope["variables"]))
    
    if result is None:
        msg = "no entry found for variable '{}'.".format(variable_tag)
        if ERROR_HANDLING  == "strict":
            utils.logging.log_error(LOG_PREFIX,"search_index_for_variable",msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logging.log_warning(LOG_PREFIX,"search_index_for_variable",msg) 
        return EMPTY_VARIABLE, False
    else:
        utils.logging.log_debug2(LOG_PREFIX,"search_index_for_variable",\
          "entry found for variable '{}'".format(variable_tag)) 
        utils.logging.log_leave_function(LOG_PREFIX,"search_index_for_variable")
        return result, True

def search_scope_for_type(scope,type_name):
    """
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"search_scope_for_type",{"type_name":type_name})
    result = _intrnl_search_scope_for_typeOrSubprogram(scope,type_name,"types",EMPTY_TYPE)
    utils.logging.log_leave_function(LOG_PREFIX,"search_scope_for_type")
    return result

def search_scope_for_subprogram(scope,subprogram_name):
    """
    :param str subprogram_name: lower case name of the searched subprogram. Simple identifier such as 'mysubroutine'.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"search_scope_for_subprogram",{"subprogram_name":subprogram_name})
    result =  _intrnl_search_scope_for_typeOrSubprogram(scope,subprogram_name,"subprograms",EMPTY_SUBPROGRAM)
    utils.logging.log_leave_function(LOG_PREFIX,"search_scope_for_subprogram")
    return result

def search_index_for_variable(index,parent_tag,variable_tag):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    %param str variable_tag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c'. Note that all array indexing expressions must be stripped away.
    :see: translator.translator.create_index_search_tag_for_variable(variable_expression)
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"search_index_for_variable",\
      {"parent_tag":parent_tag,"variable_tag":variable_tag})

    scope = create_scope(index,parent_tag)
    return search_scope_for_variable(scope,variable_tag)

def search_index_for_type(index,parent_tag,type_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str type_name: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"search_index_for_type",\
      {"parent_tag":parent_tag,"type_name":type_name})
    result = _intrnl_search_index_for_typeOrSubprogram(index,parent_tag,type_name,"types",EMPTY_TYPE)
    utils.logging.log_leave_function(LOG_PREFIX,"search_index_for_type")
    return result

def search_index_for_subprogram(index,parent_tag,subprogram_name):
    """
    :param str parent_tag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str subprogram_name: lower case name of the searched subprogram. Simple identifier such as 'mysubroutine'.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"search_index_for_subprogram",\
      {"parent_tag":parent_tag,"subprogram_name":subprogram_name})
    result =  _intrnl_search_index_for_typeOrSubprogram(index,parent_tag,subprogram_name,"subprograms",EMPTY_SUBPROGRAM)
    utils.logging.log_leave_function(LOG_PREFIX,"search_index_for_subprogram")
    return result
            
def index_variable_is_on_device(ivar):
    return ["device"] in ivar["qualifiers"] or\
           ivar["declare_on_target"] in ["alloc","to","from","tofrom"]