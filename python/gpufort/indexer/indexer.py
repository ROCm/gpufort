# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import os, sys, subprocess
import copy
import re
import concurrent.futures

import json

import pyparsing

from gpufort import util
from gpufort import translator
from gpufort import linemapper

from . import opts
from . import types
from . import grammar

considered_constructs = [
  "program", "module", "subroutine", "function",
  "type", "interface"
]

ignored_constructs = [
  "associate",
  "block",
  "case",
  "do",
  "forall",
  "if",
  "select",
  #"type",
  "where",
  "while",
  
  #"interface",
]

class Node():

    def __init__(self, kind, name, data, parent=None):
        self._kind = kind
        self._name = name
        self._parent = parent
        self._data = data
        self._begin = (-1, -1) # first linemap no, first statement no

    def __str__(self):
        return "{}: {}".format(self._name, self._data)

    def statements(self, linemaps, end):
        assert self._begin[0] > -1
        assert self._begin[1] > -1
        assert end[0] > -1
        assert end[1] > -1
        linemaps_of_node = linemaps[self._begin[0]:end[0]
                                    + 1] # upper bound is exclusive
        return [
            stmt["body"] for stmt in linemapper.get_linemaps_content(
                linemaps_of_node, "statements", self._begin[1], end[1])
        ]

    __repr__ = __str__

def create_index_records_from_declaration(statement,file_path,lineno):
    """:raise util.errorSyntaxError: If the syntax of the declaration statement is not
                                     as expected.
    """
    f_type, f_len, kind, params, qualifiers, dimension_bounds, variables, f_type_full, _ = util.parsing.parse_declaration(statement.lower())
    # TODO inconsistent, len overwrites character len but bounds do not overwrite dimension_bounds
    context = []
    for var in variables:
        name, bounds, rhs = var
        if ("parameter" in qualifiers 
           and (rhs == None or not len(rhs))):
            raise util.error.SyntaxError("parameter must have right-hand side expression")
        bounds_to_pass = bounds if len(bounds) else dimension_bounds 
        ivar = types.create_index_var(f_type, f_len, kind, params, name, qualifiers, bounds_to_pass, rhs, file_path, lineno)
        context.append(ivar)
    return context

def create_index_record_from_use_statement(statement):
    module, qualifiers, renamings, only = util.parsing.parse_use_statement(statement)
    
    used_module = {}
    used_module["name"] = module 
    used_module["attributes"] = qualifiers
    used_module["renamings"] = []
    for pair in renamings:
        used_module["renamings"].append({
            "original": pair[1],
           "renamed": pair[0],
        })
    used_module["only"] = []
    for pair in only:
        used_module["only"].append({
            "original": pair[1],
           "renamed": pair[0],
        })
    return used_module

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_fortran_construct_record(kind, name, file_path):
    entry = {}
    entry["kind"] = kind
    entry["name"] = name
    #entry["file"]        = file_path
    entry["variables"] = []
    entry["types"] = []
    entry["procedures"] = []
    entry["used_modules"] = []
    if kind in ["module","program","subroutine","function"]:
        entry["implicit"] = []
    return entry

@util.logging.log_entry_and_exit(opts.log_prefix)
def _parse_statements(linemaps, file_path,**kwargs):
    modern_fortran,_ = util.kwargs.get_value("modern_fortran",opts.modern_fortran,**kwargs)
    cuda_fortran  ,_ = util.kwargs.get_value("cuda_fortran",opts.cuda_fortran,**kwargs)
    openacc       ,_ = util.kwargs.get_value("openacc",opts.openacc,**kwargs)
    
    default_implicit_spec =\
      util.parsing.parse_implicit_statement(
        "IMPLICIT integer (i-n), real (a-h,o-z)")
    
    index = []
    root = Node("root", "root", data=index, parent=None)
    current_node = root
    current_statement = None
    accessibility_statement_stack = []
    implicit_spec_stack = []
    in_interface        = False
    interface_name      = None
    in_contains_section_stack   = [True] # file root can contain subroutines but not interfaces

    def log_enter_node_():
        nonlocal current_node
        nonlocal current_statement
        util.logging.log_debug(opts.log_prefix,"_parse_statements","[current-node={0}:{1}] enter {2} '{3}' in statement: '{4}'".format(\
          current_node._parent._kind,current_node._parent._name,
          current_node._kind,current_node._name,\
          current_statement))

    def log_leave_node_():
        nonlocal current_node
        nonlocal current_statement
        util.logging.log_debug(opts.log_prefix,"_parse_statements","[current-node={0}:{1}] leave {0} '{1}' in statement: '{2}'".format(\
          current_node._data["kind"],current_node._data["name"],\
          current_statement))

    def log_detection_(kind):
        nonlocal current_node
        nonlocal current_statement
        util.logging.log_debug2(opts.log_prefix,"_parse_statements","[current-node={}:{}] found {} in statement: '{}'".format(\
                current_node._kind,current_node._name,kind,current_statement))
    
    def log_begin_task(parent_node, msg):
        util.logging.log_debug3(opts.log_prefix,"_parse_statements","[parent-node={0}:{1}] {2}".format(
              parent_node._kind, parent_node._name, msg))

    def log_end_task(parent_node, msg):
        util.logging.log_debug2(opts.log_prefix,"_parse_statements","[parent-node={0}:{1}] {2}".format(
              parent_node._kind, parent_node._name, msg))

    def get_current_implicit_rules_():
        if not len(implicit_spec_stack[-1]):
            return default_implicit_spec
        else:
            return implicit_spec_stack[-1]

    # Parser events
    # direct parsing
    def End():
        nonlocal root
        nonlocal current_node
        nonlocal linemaps
        nonlocal current_linemap_no
        nonlocal current_statement_no
        nonlocal current_statement
        nonlocal current_tokens
        nonlocal accessibility_statement_stack
        nonlocal implicit_spec_stack
        log_detection_("end of program/module/subroutine/function/type")
        if current_node._kind != "root":
            log_leave_node_()
            accelerator_routine = False
            if current_node._kind in ["subroutine","function"] and\
               len(current_node._data["attributes"]):
                for q in current_node._data["attributes"]:
                    if q == "global" or "device" in q:
                        accelerator_routine = True
            if current_node._kind == "type" or accelerator_routine:
                end = (current_linemap_no, current_statement_no)
                current_node._data["statements"] = current_node.statements(
                    linemaps, end)
            # implicit rules: 
            # (assumes only 1 rule if first rule is implicit none)
            # (assumes redundant usage of same letter is not happening)
            # (Implicit() function performs the necessary checks)
            if current_node._kind in ["module","program","subroutine","function"]:
                for f_type,f_len,kind,letters in get_current_implicit_rules_():
                    if len(letters):
                        current_node._data["implicit"].append({
                          "f_type":f_type,"len":f_len,"kind":kind,
                          "letters": letters
                        })
                implicit_spec_stack.pop(-1)
                in_contains_section_stack.pop(-1)
            # accessibility: public vs private
            if current_node._kind in ["module","type"]:
                # apply public/private visibility
                general_accessibility = "public"
                modes = ["public","private"]
                explicitly_set_accessibility = {"public": [], "private": []}
                for kind,identifiers in accessibility_statement_stack.pop(-1):
                    if not len(identifiers):
                        general_accessibility = kind
                    explicitly_set_accessibility[kind] += identifiers
                current_node._data["accessibility"] = general_accessibility
                current_node._data["public"] = explicitly_set_accessibility["public"] 
                current_node._data["private"] = explicitly_set_accessibility["private"]
            current_node = current_node._parent

    def ModuleStart():
        nonlocal root
        nonlocal current_node
        nonlocal current_tokens
        nonlocal accessibility_statement_stack
        nonlocal implicit_spec_stack
        nonlocal in_contains_section_stack
        accessibility_statement_stack.append([])
        implicit_spec_stack.append([])
        in_contains_section_stack.append(False)
        #
        name = current_tokens[1]
        module = create_fortran_construct_record("module", name, file_path)
        module["accessibility"] = opts.default_module_accessibility
        module["public"] = []
        module["private"] = []
        assert current_node == root
        current_node._data.append(module)
        current_node = Node("module", name, data=module, parent=current_node)
        log_enter_node_()

    def ProgramStart():
        nonlocal root
        nonlocal current_node
        nonlocal current_tokens
        nonlocal implicit_spec_stack
        nonlocal in_contains_section_stack
        implicit_spec_stack.append([])
        in_contains_section_stack.append(False)
        #
        name = current_tokens[1]
        program = create_fortran_construct_record("program", name, file_path)
        assert current_node._kind == "root"
        current_node._data.append(program)
        current_node = Node("program", name, data=program, parent=current_node)
        log_enter_node_()


    def PublicOrPrivate():
        nonlocal current_node
        nonlocal current_tokens
        if current_node._kind in ["module","type"]:
            kind, identifiers = util.parsing.parse_public_or_private_statement(
              current_statement,current_tokens[0])
            for entry in accessibility_statement_stack[-1]:
                other_kind, other_identifiers = entry
                if not len(identifiers) and not len(other_identifiers):
                    raise util.error.SyntaxError("'{}' two 'private' or 'public' statements without symbol list in same module")
                identifiers_subject_to_other_statement = [expr for expr in identifiers if expr in other_identifiers]
                if len(identifiers_subject_to_other_statement):
                    raise util.error.SyntaxError("symbols '{}' have already been subject to another "+
                                            "'private' or 'public' statement with symbol list".format(
                                                "','".join(identifiers_subject_to_other_statement)))
            accessibility_statement_stack[-1].append((kind,identifiers))
        else:
            raise util.error.SyntaxError("unexpected '{}' statement".format(current_tokens[0]))
    
    def Interface():
        nonlocal in_interface
        nonlocal interface_name
        nonlocal current_statement
        log_detection_("start of interface")
        if in_contains_section_stack[-1]:
            raise util.error.SyntaxError("interface statement in contains section")
        if in_interface:
            raise util.error.SyntaxError("interface statement within interface construct")
        in_interface = True
        name = util.parsing.parse_interface_statement(current_statement)
        interface_name = name
    
    def InterfaceEnd():
        nonlocal in_interface
        nonlocal interface_name
        log_detection_("end of interface")
        in_interface   = False
        interface_name = None
    
    def Contains():
        nonlocal in_contains_section_stack
        log_detection_("contains")
        in_contains_section_stack[-1] = True
        
        
    def Implicit():
        nonlocal current_node
        nonlocal current_tokens
        nonlocal current_statement
        log_detection_("implicit")
        if current_node._kind in ["module","program","subroutine","function"]:
            new_specs = util.parsing.parse_implicit_statement(current_statement)
            for _,_,_,letters in new_specs:
                for _,_,_,other_letters in implicit_spec_stack[-1]:
                    first_mutual_letter = next((l for l in letters if l in other_letters),None)
                    if first_mutual_letter != None:
                        raise util.error.SyntaxError("more than one implicit rule found for letter '{}'".format(first_mutual_letter))
            implicit_spec_stack[-1] += new_specs
        else:
            raise util.error.SyntaxError("unexpected 'implicit' statement")
        pass

    #host|device,name,[args]
    def SubroutineStart():
        nonlocal current_node
        nonlocal current_statement
        nonlocal implicit_spec_stack
        nonlocal in_contains_section_stack
        nonlocal interface_name
        implicit_spec_stack.append([])
        #
        log_detection_("start of subroutine")
        if current_node._kind in [
                "root", "module", "program", "subroutine", "function"
        ]:
            kind, name, dummy_args, modifiers, attributes,\
            result_triple, bind_tuple = util.parsing.parse_function_statement(current_statement)
            result_type, result_type_kind, result_name = result_triple
            bind_c, bind_c_name = bind_tuple
            subroutine = create_fortran_construct_record("subroutine", name, file_path)
            subroutine["attributes"]     = modifiers + attributes 
            subroutine["interface"]      = not in_contains_section_stack[-1]
            subroutine["interface_name"] = interface_name # TODO consider module procedure & check if multiple names can be used
            subroutine["dummy_args"] = dummy_args
            if current_node._kind == "root":
                current_node._data.append(subroutine)
            else:
                current_node._data["procedures"].append(subroutine)
            current_node = Node("subroutine",
                                name,
                                data=subroutine,
                                parent=current_node)
            current_node._begin = (current_linemap_no, current_statement_no)
            log_enter_node_()
            in_contains_section_stack.append(False)
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found subroutine in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))

    #host|device,name,[args],result
    def FunctionStart():
        nonlocal current_statement
        nonlocal current_node
        nonlocal current_linemap
        nonlocal implicit_spec_stack
        nonlocal in_contains_section_stack
        nonlocal interface_name
        implicit_spec_stack.append([])
        #
        log_detection_("start of function")
        if current_node._kind in [
                "root", "module", "program", "subroutine", "function"
        ]:
            kind, name, dummy_args, modifiers, attributes,\
            result_triple, bind_tuple = util.parsing.parse_function_statement(current_statement)
            result_type, result_type_kind, result_name = result_triple
            bind_c, bind_c_name = bind_tuple
            function = create_fortran_construct_record("function", name, file_path)
            function["attributes"]     = modifiers + attributes 
            function["interface"]      = not in_contains_section_stack[-1]
            function["interface_name"] = interface_name # TODO add name here
            function["dummy_args"]     = dummy_args
            function["result_name"]    = result_name
            if result_type != None:
                result_var_decl = result_type
                if result_type_kind != None:
                    result_var_decl += "(" + result_type_kind + ")"
                result_var_decl += " "+result_name
                function["variables"] += create_index_records_from_declaration(result_var_decl,
                                                                               current_linemap["file"],
                                                                               current_linemap["lineno"])
            if current_node._kind == "root":
                current_node._data.append(function)
            else:
                current_node._data["procedures"].append(function)
            current_node = Node("function",
                                name,
                                data=function,
                                parent=current_node)
            current_node._begin = (current_linemap_no, current_statement_no)
            log_enter_node_()
            in_contains_section_stack.append(False)
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found function in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))

    def TypeStart():
        nonlocal current_linemap
        nonlocal current_linemap_no
        nonlocal current_statement
        nonlocal current_statement_no
        nonlocal current_node
        nonlocal accessibility_statement_stack
        log_detection_("start of type")
        if current_node._kind in [
                "module", "program", "subroutine", "function"
        ]:
            name,attributes,params = util.parsing.parse_type_statement(
                    current_statement)
            derived_type = {}
            derived_type["name"] = name
            derived_type["kind"] = "type"
            derived_type["attributes"] = attributes
            derived_type["accessibility"] = opts.default_type_accessibility
            derived_type["public"] = []
            derived_type["private"] = []
            derived_type["params"] = params
            derived_type["variables"] = []
            derived_type["types"] = []
            derived_type["file"] = current_linemap["file"]
            derived_type["lineno"] = current_linemap["lineno"]  
            current_node._data["types"].append(derived_type)
            current_node = Node("type",
                                name,
                                data=derived_type,
                                parent=current_node)
            current_node._begin = (current_linemap_no, current_statement_no)
            accessibility_statement_stack.append([])
            log_enter_node_()
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found derived type in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
                    format(current_statement,current_node._kind))

    def Use():
        nonlocal current_node
        nonlocal current_statement
        log_detection_("use statement")
        
        if current_node._kind != "root":
            current_node._data["used_modules"].append(
                create_index_record_from_use_statement(current_statement))

    # delayed parsing
    def Declaration():
        nonlocal root
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement
        nonlocal current_statement_no
        #print(current_statement)
        log_detection_("declaration")
        if current_node != root:
            msg = "begin to parse declaration '{}'".format(
                current_statement)
            log_begin_task(current_node, msg)
            variables = create_index_records_from_declaration(current_statement,
                                                              current_linemap["file"],
                                                              current_linemap["lineno"])
            current_node._data["variables"] += variables
            msg = "finished to parse declaration '{}'".format(current_statement)
            log_end_task(current_node, msg)
    
    def Parameter():
        """Add parameter attribute to previously declared variables in same scope/declaration list.
        Does not modify variables in other scopes.
        """
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        # TODO need to parse implicit statements too
        log_detection_("parameter")
        if current_node != root:
            msg = "begin to parse parameter statement '{}'".format(
                current_statement)
            log_begin_task(current_node, msg)
            #
            parameters = util.parsing.parse_parameter_statement(current_statement)
            for pair in parameters:
                var_name, rhs = pair 
                found_decl = False
                for ivar in current_node._data["variables"]:
                    if ivar["name"] == var_name:
                        found_decl = True
                        if "parameter" in ivar["attributes"]:
                            raise util.error.SyntaxError("variable '{}' has already 'parameter' attribute".format(var_name))
                        ivar["attributes"].append("parameter")
                        if ivar["rhs"] != None and len(ivar["rhs"]):
                            raise util.error.SyntaxError("variable '{}' is already initialized".format(var_name))
                        ivar["rhs"] = rhs
                if not found_decl:
                    #print(get_current_implicit_rules_())
                    for f_type,f_len,kind,letters in get_current_implicit_rules_():
                        if var_name[0] in letters:
                            found_decl = True
                            ivar = types.create_index_var(f_type, f_len, kind, [], var_name, ["parameter"], [], rhs, 
                              current_linemap["file"], current_linemap["lineno"])
                            current_node._data["variables"].append(ivar)
                            break
                if not found_decl:
                    raise util.error.SyntaxError("no declaration or applicable implicit specification found for parameter '{}'".format(var_name))
            #
            msg = "finished to parse parameter statement '{}'".format(current_statement)
            log_end_task(current_node, msg)

    def Attributes():
        """Add attributes to previously declared variables in same scope/declaration list.
        Does not modify variables in other scopes.
        """
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        #print(current_statement)
        log_detection_("attributes statement")
        if current_node != root:
            msg = "begin to parse attributes statement '{}'".format(
                current_statement)
            log_begin_task(current_node, msg)
            #
            attributes, modified_vars = util.parsing.parse_attributes_statement(current_statement)
            for var_context in current_node._data["variables"]:
                if var_context["name"] in modified_vars:
                    var_context["attributes"] += attributes
            #
            msg = "finished to parse attributes statement '{}'".format(current_statement)
            log_end_task(current_node, msg)

    def AccDeclare():
        """Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        log_detection_("acc declare directive")
        msg = "begin to parse acc declare directive '{}'".format(
            current_statement)
        log_begin_task(current_node, msg)
        #
        parse_result = translator.tree.grammar.acc_declare.parseString( # TODO switch to token based parser
            current_statement)[0]
        for var_context in current_node._data["variables"]:
            for var_name in parse_result.map_alloc_vars():
                if var_context["name"] == var_name:
                    var_context["declare_on_target"] = "alloc"
            for var_name in parse_result.map_to_vars():
                if var_context["name"] == var_name:
                    var_context["declare_on_target"] = "to"
            for var_name in parse_result.map_from_vars():
                if var_context["name"] == var_name:
                    var_context["declare_on_target"] = "from"
            for var_name in parse_result.map_tofrom_vars():
                if var_context["name"] == var_name:
                    var_context["declare_on_target"] = "tofrom"
        msg = "finished to parse acc declare directive '{}'".format(current_statement)
        log_end_task(current_node, msg)

    def AccRoutine():
        """Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        log_detection_("acc routine directive")
        if current_node != root:
            msg = "begin to parse acc routine directive '{}'".format(
                current_statement)
           # parse_result = translator.tree.grammar.acc_routine.parseString( # TODO switch to token based parser
           #     current_statement)[0]
            _, _, directive_args, clauses = util.parsing.parse_acc_directive(
                    current_statement)
            if len(clauses) != 1:
                raise util.error.SyntaxError("expected one of 'gang', 'worker', 'vector', 'seq'")
            if not len(directive_args): # TODO len(directive_args) applies to external or included routine
                parallelism = clauses[0]
                if parallelism == "seq":
                    current_node._data["attributes"] += ["host", "device"]
                elif parallelism == "gang":
                    current_node._data["attributes"] += ["host", "device:gang"]
                elif parallelism == "worker":
                    current_node._data["attributes"] += ["host", "device:worker"]
                elif parallelism == "vector":
                    current_node._data["attributes"] += ["host", "device:vector"]
                else:
                    raise util.error.SyntaxError("expected one of 'gang', 'worker', 'vector', 'seq'")
            msg = "finished to parse acc routine directive '{}'".format(current_statement)
            log_end_task(current_node, msg)

    # parser loop
    try:
        for current_linemap_no, current_linemap in enumerate(linemaps):
            if current_linemap["is_active"]:
                for current_statement_no, stmt in enumerate(
                        current_linemap["statements"]):
                    original_statement_lower = stmt["body"].lower()
                    current_tokens = util.parsing.tokenize(original_statement_lower,padded_size=6)
                    try:
                        numeric_label = str(int(current_tokens[0]))
                        current_tokens.pop(0)
                    except:
                        numeric_label = None
                    current_statement = " ".join([tk for tk in current_tokens if len(tk)])
                    util.logging.log_debug3(
                        opts.log_prefix, "_parse_statements",
                        "process statement '{}'".format(current_statement))
                    if openacc and util.parsing.is_fortran_directive(original_statement_lower,modern_fortran):
                        if current_tokens[1:3] == ["acc","declare"]:
                            AccDeclare()
                        elif current_tokens[1:3] == ["acc","routine"]:
                            AccRoutine()
                    elif util.parsing.is_fortran_comment(original_statement_lower,modern_fortran):
                        pass
                    else: # fortran statements
                        if util.parsing.is_assignment(current_tokens):
                            # Fortran allows that variables can be named after keywords, e.g. 'function', 'real'.
                            # We must exclude that a keyword-named variable
                            # appears in an assignment before trying to detect
                            # another Fortran statement. Hence we first check if a statement
                            # is an assignment.
                            # TODO could handle statement functions here 
                            # TODO handle implicit variables here (problem: scope)
                            pass
                        else:
                            if current_tokens[0:2] == ["end","interface"]:
                                InterfaceEnd() 
                            elif (current_tokens[0]=="end" and
                               current_tokens[1] not in ignored_constructs):
                                End()
                            elif openacc and util.parsing.is_fortran_directive(original_statement_lower,modern_fortran):
                                if current_tokens[1:3] == ["acc","declare"]:
                                    AccDeclare()
                                elif current_tokens[1:3] == ["acc","routine"]:
                                    AccRoutine()
                            elif current_tokens[0] == "use":
                                Use()
                            #elif current_tokens[0] == "implicit":
                            #    try_to_parse_string("implicit",grammar.IMPLICIT)
                            elif current_tokens[0] == "module" and current_tokens[1] != "procedure":
                                ModuleStart()
                            elif current_tokens[0] == "program":
                                ProgramStart()
                            elif current_tokens[0] == "type" and current_tokens[1] != "(": # type a ; type, bind(c) :: a
                                TypeStart()
                            elif current_tokens[0] in ["public","private"]:
                                PublicOrPrivate()
                            elif current_tokens[0] == "implicit":
                                Implicit()
                            elif current_tokens[0] == "interface":
                                Interface()
                            elif current_tokens[0] == "contains":
                                Contains()
                            elif current_tokens[0] == "parameter":
                                Parameter()
                            elif current_tokens[0] == "attributes" and "::" in current_tokens: # attributes(device) :: a
                                Attributes() 
                            # TODO parse functions, subroutine signatures more carefully
                            elif current_tokens[0] != "end" and "function" in current_tokens:
                                # Functions must be checked before the 
                                FunctionStart()
                            elif current_tokens[0] != "end" and "subroutine" in current_tokens:
                                SubroutineStart()
                            elif current_tokens[0] in [
                                 "character", "integer", "logical", "real",
                                 "complex", "double"
                               ]:
                                Declaration()
                            elif current_tokens[0:2] == ["type","("]: # type(dim3) :: a
                                Declaration()
    except util.error.SyntaxError as e:
        file_path = current_linemap["file"]
        lineno = current_linemap["lineno"]
        msg = "{}:{}:{}(stmt-no):{}".format(file_path,lineno,current_statement_no+1,e.args[0])
        e.args = (msg,)
        raise

    return index


@util.logging.log_entry_and_exit(opts.log_prefix)
def _write_json_file(index, file_path):
    with open(file_path, "w") as outfile:
        json.dump(index,outfile)

@util.logging.log_entry_and_exit(opts.log_prefix)
def _read_json_file(file_path):
    util.logging.log_debug2(opts.log_prefix,"_read_json_file","".join(["reading file:",file_path]))
    #print(file_path,file=sys.stderr)
    with open(file_path, "r") as infile:
        return json.load(infile)

# API
@util.logging.log_entry_and_exit(opts.log_prefix)
def update_index_from_linemaps(linemaps, index,**kwargs):
    """Updates index from a number of linemaps."""
    if len(linemaps):
        index += _parse_statements(linemaps,
                                   file_path=linemaps[0]["file"],
                                   **kwargs)

@util.logging.log_entry_and_exit(opts.log_prefix)
def update_index_from_snippet(index, snippet, **kwargs):
    linemaps = linemapper.preprocess_and_normalize(snippet.splitlines(),
                                                   file_path="dummy.f90", **kwargs)
    update_index_from_linemaps(linemaps, index, **kwargs)


@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index_from_snippet(snippet, **kwargs):
    index = []
    update_index_from_snippet(index, snippet, **kwargs)
    return index

@util.logging.log_entry_and_exit(opts.log_prefix)
def search_derived_types(imodule, search_procedures=True):
    """Search through a module (or program)
    and return derived type definition index entries.
    """

    util.logging.log_enter_function(opts.log_prefix,"_search_derived_types",\
      {"modulename": imodule.get("name",""),\
       "search_procedures": str(search_procedures)})

    itypes = {}
    prefix = ""

    def collect_types_(irecord):
        nonlocal itypes
        nonlocal prefix
        prefix += irecord["name"]
        for itype in irecord["types"]:
            ident = prefix + "_" + itype["name"]
            itypes[ident] = itype
        if search_procedures:
            for iprocedure in irecord["procedures"]:
                collect_types_(iprocedure)

    collect_types_(imodule)

    util.logging.log_leave_function(
        opts.log_prefix, "_search_derived_types",
        {"typenames": str([itype["name"] for itype in itypes.values()])})

    return itypes


@util.logging.log_entry_and_exit(opts.log_prefix)
def write_gpufort_module_files(index, output_dir):
    """
    Per module / program found in the index
    write a GPUFORT module file.
    :param list index:    [in] Empty or non-empty list.
    :param str output_dir: [in] Output directory.
    """
    for mod in index:
        file_path = output_dir + "/" + mod["name"] + opts.gpufort_module_file_suffix
        _write_json_file(mod, file_path)


@util.logging.log_entry_and_exit(opts.log_prefix)
def load_gpufort_module_files(input_dirs, index):
    """
    Load gpufort module files and append to the index.

    :param list input_dirs: [in] List of input directories (as strings).
    :param list index:     [inout] Empty or non-empty list. Loaded data structure is appended.
    """
    for input_dir in input_dirs:
        for child in os.listdir(input_dir):
            if child.endswith(opts.gpufort_module_file_suffix):
                module_already_exists = False
                for mod in index:
                    if mod == child.replace(opts.gpufort_module_file_suffix, ""):
                        module_already_exists = True
                        break
                if not module_already_exists:
                    mod_index = _read_json_file(
                        os.path.join(input_dir, child))
                    index.append(mod_index)
