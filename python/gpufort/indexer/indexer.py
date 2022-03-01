# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import os, sys, subprocess
import copy
import re
import concurrent.futures

import orjson

from gpufort import util
from gpufort import translator
from gpufort import linemapper
from . import opts

GPUFORT_MODULE_FILE_SUFFIX = ".gpufort_mod"

CASELESS = False
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), "../grammar")
exec(open(os.path.join(GRAMMAR_DIR, "grammar.py")).read())

# configurable parameters
p_filter = re.compile(opts.filter)
p_continuation = re.compile(opts.continuation_filter)

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

def create_index_records_from_declaration(statement):
    """:raise SyntaxError: If the syntax of the declaration statement is not
                           as expected.
    """
    f_type, kind, qualifiers, dimension_bounds, variables, _, __ = util.parsing.parse_declaration(statement.lower())
    context = []
    for var in variables:
        name, bounds, rhs = var
        ivar = {}
        # basic
        ivar["name"]   = name
        ivar["f_type"] = f_type
        ivar["kind"]   = kind
        # TODO bytes per element can be computed on the fly
        ivar["bytes_per_element"] = translator.num_bytes(f_type, kind, default=None)
        ivar["c_type"]            = translator.convert_to_c_type(f_type, kind, "TODO unknown")
        ivar["qualifiers"] = qualifiers
        # ACC/OMP
        ivar["declare_on_target"] = False
        # arrays
        ivar["bounds"] = bounds + dimension_bounds
        ivar["rank"]   = len(ivar["bounds"])
        # handle parameters
        #ivar["value"] = None # TODO parse rhs if necessary
        ivar["rhs"] = rhs
        context.append(ivar)
    return context

@util.logging.log_entry_and_exit(opts.log_prefix)
def _parse_statements(linemaps, file_path):
    # Regex
    datatype_reg = Regex(
        r"\b(type\s*\(|character|integer|logical|real|complex|double\s+precision)\b"
    )

    index = []

    # statistics
    total_num_tasks = 0

    def log_begin_task(parent_node, msg):
        util.logging.log_debug3(opts.log_prefix,"_parse_statements","[parent-node={0}:{1}] {2}".format(
              parent_node._kind, parent_node._name, msg))

    def log_end_task(parent_node, msg):
        util.logging.log_debug2(opts.log_prefix,"_parse_statements","[parent-node={0}:{1}] {2}".format(
              parent_node._kind, parent_node._name, msg))
    
    # Parser events
    root = Node("root", "root", data=index, parent=None)
    current_node = root
    current_statement = None

    def create_base_entry_(kind, name, file_path):
        entry = {}
        entry["kind"] = kind
        entry["name"] = name
        #entry["file"]        = file_path
        entry["variables"] = []
        entry["types"] = []
        entry["procedures"] = []
        entry["used_modules"] = []
        return entry

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

    # direct parsing
    def End():
        nonlocal root
        nonlocal current_node
        nonlocal linemaps
        nonlocal current_linemap_no
        nonlocal current_statement_no
        nonlocal current_statement
        log_detection_("end of program/module/subroutine/function")
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
            current_node = current_node._parent

    def ModuleStart(tokens):
        nonlocal root
        nonlocal current_node
        name = tokens[0]
        module = create_base_entry_("module", name, file_path)
        assert current_node == root
        current_node._data.append(module)
        current_node = Node("module", name, data=module, parent=current_node)
        log_enter_node_()

    def ProgramStart(tokens):
        nonlocal root
        nonlocal current_node
        name = tokens[0]
        program = create_base_entry_("program", name, file_path)
        assert current_node._kind == "root"
        current_node._data.append(program)
        current_node = Node("program", name, data=program, parent=current_node)
        log_enter_node_()

    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal current_statement
        nonlocal current_node
        log_detection_("start of subroutine")
        if current_node._kind in [
                "root", "module", "program", "subroutine", "function"
        ]:
            name = tokens[1]
            subroutine = create_base_entry_("subroutine", name, file_path)
            subroutine["attributes"] = [q.lower() for q in tokens[0]]
            subroutine["dummy_args"] = list(tokens[2])
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
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found subroutine in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))

    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal current_statement
        nonlocal current_node
        log_detection_("start of function")
        if current_node._kind in [
                "root", "module", "program", "subroutine", "function"
        ]:
            name = tokens[1]
            function = create_base_entry_("function", name, file_path)
            function["attributes"] = [q.lower() for q in tokens[0]]
            function["dummy_args"] = list(tokens[2])
            function["result_name"] = name if tokens[3] is None else tokens[3]
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
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found function in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))

    def TypeStart(tokens):
        nonlocal current_linemap_no
        nonlocal current_statement
        nonlocal current_statement_no
        nonlocal current_node
        log_detection_("start of type")
        if current_node._kind in [
                "module", "program", "subroutine", "function"
        ]:
            assert len(tokens) == 2
            name = tokens[1]
            derived_type = {}
            derived_type["name"] = name
            derived_type["kind"] = "type"
            derived_type["variables"] = []
            derived_type["types"] = []
            current_node._data["types"].append(derived_type)
            current_node = Node("type",
                                name,
                                data=derived_type,
                                parent=current_node)
            current_node._begin = (current_linemap_no, current_statement_no)
            log_enter_node_()
        else:
            util.logging.log_warning(opts.log_prefix,"_parse_statements","found derived type in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
                    format(current_statement,current_node._kind))

    def Use(tokens):
        nonlocal current_node
        log_detection_("use statement")
        if current_node._kind != "root":
            used_module = {}
            used_module["name"] = translator.tree.make_f_str(tokens[1])
            used_module["only"] = []
            for pair in tokens[2]:
                original = translator.tree.make_f_str(pair[0])
                renamed = original if pair[
                    1] is None else translator.tree.make_f_str(pair[1])
                used_module["only"].append({
                    "original": original,
                    "renamed": renamed
                })
            current_node._data["used_modules"].append(
                used_module) # TODO only include what is necessary

    # delayed parsing
    def Declaration(tokens):
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        nonlocal total_num_tasks
        #print(current_statement)
        log_detection_("declaration")
        if current_node != root:
            total_num_tasks += 1
            msg = "begin to parse declaration '{}'".format(
                current_statement)
            log_begin_task(current_node, msg)
            variables = create_index_records_from_declaration(current_statement)
            current_node._data["variables"] += variables
            msg = "finished to parse declaration '{}'".format(current_statement)
            log_end_task(current_node, msg)

    def Attributes(tokens):
        """Add attributes to previously declared variables in same scope/declaration list.
        Does not modify scope of other variables.
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
            attribute, modified_vars = \
                translator.parse_attributes(translator.tree.grammar.attributes.parseString(current_statement)[0])
            for var_context in current_node._data["variables"]:
                if var_context["name"] in modified_vars:
                    var_context["qualifiers"].append(attribute)
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
        parse_result = translator.tree.grammar.acc_declare.parseString(
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
            parse_result = translator.tree.grammar.acc_routine.parseString(
                current_statement)[0]
            if parse_result.parallelism() == "seq":
                current_node._data["attributes"] += ["host", "device"]
            elif parse_result.parallelism() == "gang":
                current_node._data["attributes"] += ["host", "device:gang"]
            elif parse_result.parallelism() == "worker":
                current_node._data["attributes"] += ["host", "device:worker"]
            elif parse_result.parallelism() == "vector":
                current_node._data["attributes"] += ["host", "device:vector"]
            msg = "finished to parse acc routine directive '{}'".format(current_statement)
            log_end_task(current_node, msg)

    module_start.setParseAction(ModuleStart)
    type_start.setParseAction(TypeStart)
    program_start.setParseAction(ProgramStart)
    function_start.setParseAction(FunctionStart)
    subroutine_start.setParseAction(SubroutineStart)

    type_end.setParseAction(End)
    structure_end.setParseAction(End)

    datatype_reg.setParseAction(Declaration)
    use.setParseAction(Use)
    attributes.setParseAction(Attributes)

    def try_to_parse_string(expression_name, expression):
        try:
            expression.parseString(current_statement)
            return True
        except ParseBaseException as e:
            util.logging.log_debug3(
                opts.log_prefix, "_parse_statements",
                "did not find expression '{}' in statement '{}'".format(
                    expression_name, current_statement))
            util.logging.log_debug4(opts.log_prefix,
                                    "_parse_statements", str(e))
            return False

    def is_end_statement_(tokens, kind):
        result = tokens[0] == "end" + kind
        if not result and len(tokens):
            result = tokens[0] == "end" and tokens[1] == kind
        return result

    def consider_statement(stripped_statement):
        passes_filter = p_filter.match(stripped_statement) != None
        return passes_filter

    # parser loop
    for current_linemap_no, current_linemap in enumerate(linemaps):
        if current_linemap["is_active"]:
            for current_statement_no, stmt in enumerate(
                    current_linemap["statements"]):
                current_statement = stmt["body"].lower().strip(" \t\n")
                if not consider_statement(current_statement):
                    util.logging.log_debug3(
                        opts.log_prefix, "_collect_statements",
                        "ignore statement '{}'".format(current_statement))
                else:
                    util.logging.log_debug3(
                        opts.log_prefix, "_parse_statements",
                        "process statement '{}'".format(current_statement))
                    current_tokens = re.split(
                        r"\s+|\t+",
                        current_statement.lower().strip(" \t"))
                    current_statement_stripped = "".join(current_tokens)
                    for expr in [
                            "program", "module", "subroutine", "function",
                            "type"
                    ]:
                        if is_end_statement_(current_tokens, expr):
                            End()
                    for comment_char in "!*c":
                        if current_tokens[0] == comment_char + "$acc":
                            if current_tokens[1] == "declare":
                                AccDeclare()
                            elif current_tokens[1] == "routine":
                                AccRoutine()
                    if current_tokens[0] == "use":
                        try_to_parse_string("use", use)
                    #elif current_tokens[0] == "implicit":
                    #    try_to_parse_string("implicit",IMPLICIT)
                    elif current_tokens[0] == "module":
                        try_to_parse_string("module", module_start)
                    elif current_tokens[0] == "program":
                        try_to_parse_string("program", program_start)
                    elif current_tokens[0].startswith(
                            "type"): # type a ; type, bind(c) :: a
                        try_to_parse_string("type", type_start)
                    elif current_tokens[0].startswith(
                            "attributes"): # attributes(device) :: a
                        try_to_parse_string("attributes", attributes)
                    # cannot be combined with above checks
                    if "function" in current_tokens:
                        try_to_parse_string("function", function_start)
                    elif "subroutine" in current_tokens:
                        try_to_parse_string("subroutine", subroutine_start)
                    for expr in [
                            "type", "character", "integer", "logical", "real",
                            "complex", "double"
                    ]: # type(dim3) :: a
                        if expr in current_tokens[0]:
                            try_to_parse_string("declaration", datatype_reg)
                            break
                    #try_to_parse_string("declaration|type_start|use|attributes|module_start|program_start|function_start|subroutine_start",\
                    #  datatype_reg|type_start|use|attributes|module_start|program_start|function_start|subroutine_start)

    # apply attributes and acc variable modifications
    return index


@util.logging.log_entry_and_exit(opts.log_prefix)
def _write_json_file(index, file_path):
    with open(file_path, "wb") as outfile:
        if opts.pretty_print_index_file:
            outfile.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))
        else:
            outfile.write(orjson.dumps(index))


@util.logging.log_entry_and_exit(opts.log_prefix)
def _read_json_file(file_path):
    with open(file_path, "rb") as infile:
        return orjson.loads(infile.read())


# API
@util.logging.log_entry_and_exit(opts.log_prefix)
def scan_file(file_path, preproc_options, index):
    """Creates an index from a single file.
    """
    filtered_statements = _read_fortran_file(file_path, preproc_options)
    util.logging.log_debug2(opts.log_prefix,"scan_file","extracted the following statements:\n>>>\n{}\n<<<".format(\
        "\n".join(filtered_statements)))
    index += _parse_statements(filtered_statements, file_path)


@util.logging.log_entry_and_exit(opts.log_prefix)
def update_index_from_linemaps(linemaps, index):
    """Updates index from a number of linemaps."""
    if len(linemaps):
        index += _parse_statements(linemaps,
                                          file_path=linemaps[0]["file"])


@util.logging.log_entry_and_exit(opts.log_prefix)
def update_index_from_snippet(index, snippet, preproc_options=""):
    macro_stack = linemapper.init_macros(preproc_options)
    linemaps = linemapper.preprocess_and_normalize(snippet.splitlines(),
                                                   "dummy.f90", macro_stack)
    update_index_from_linemaps(linemaps, index)


@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index_from_snippet(snippet, preproc_options=""):
    index = []
    update_index_from_snippet(index, snippet, preproc_options)
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
        file_path = output_dir + "/" + mod["name"] + GPUFORT_MODULE_FILE_SUFFIX
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
            if child.endswith(GPUFORT_MODULE_FILE_SUFFIX):
                module_already_exists = False
                for mod in index:
                    if mod == child.replace(GPUFORT_MODULE_FILE_SUFFIX, ""):
                        module_already_exists = True
                        break
                if not module_already_exists:
                    mod_index = _read_json_file(
                        os.path.join(input_dir, child))
                    index.append(mod_index)
