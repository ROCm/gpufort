# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys,subprocess
import copy
import re
import threading
import concurrent.futures

import orjson

import translator.translator as translator
import utils.logging
import linemapper.linemapperutils as linemapperutils

GPUFORT_MODULE_FILE_SUFFIX=".gpufort_mod"

CASELESS    = False
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open(os.path.join(GRAMMAR_DIR,"grammar.py")).read())

# configurable parameters
indexer_dir = os.path.dirname(__file__)
exec(open(os.path.join(indexer_dir,"indexer_options.py.in")).read())
    
p_filter       = re.compile(FILTER) 
p_continuation = re.compile(CONTINUATION_FILTER)

class Node():
    def __init__(self,kind,name,data,parent=None):
        self._kind     = kind
        self._name     = name
        self._parent   = parent 
        self._data     = data
        self._begin    = (-1,-1) # first linemap no, first statement no
    def __str__(self):
        return "{}: {}".format(self._name,self._data)
    def statements(self,linemaps,end):
        assert self._begin[0] > -1
        assert self._begin[1] > -1
        assert end[0] > -1
        assert end[1] > -1
        linemaps_of_node = linemaps[self._begin[0]:end[0]+1] # upper bound is exclusive
        return [stmt["body"] for stmt in linemapperutils.get_linemaps_content(linemaps_of_node,"statements",self._begin[1],end[1])]
    __repr__ = __str__

def _intrnl_parse_statements(linemaps,filepath):
    global LOG_PREFIX
    global PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE
    global PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE 

    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_parse_statements",{"filepath":filepath})
    # Regex
    datatype_reg = Regex(r"\b(type\s*\(|character|integer|logical|real|complex|double\s+precision)\b")

    index = []

    access_lock   = threading.Lock()
    utils.logging.log_debug(LOG_PREFIX,"_intrnl_parse_statements","create thread pool of size {} for process variable declarations".format(\
      PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE))
    task_executor = concurrent.futures.ThreadPoolExecutor(\
      max_workers=PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE)
    # statistics
    total_num_tasks = 0
 
    def log_enter_job_or_task_(parent_node,msg):
        utils.logging.log_debug3(LOG_PREFIX,"_intrnl_parse_statements","[thread-id={3}][parent-node={0}:{1}] {2}".format(\
              parent_node._kind, parent_node._name, msg,\
              threading.get_ident()))
        
    def log_leave_job_or_task_(parent_node,msg):
        utils.logging.log_debug2(LOG_PREFIX+"_intrnl_parse_statements","[thread-id={3}][parent-node={0}:{1}] {2}".format(\
              parent_node._kind, parent_node._name, msg,\
              threading.get_ident()))
    
    def ParseDeclarationTask_(parent_node,input_text):
        """
        :note: the term 'task' should highlight that this is a function
        that is directly submitted to a thread in the worker pool.
        """
        global LOG_PREFIX
        nonlocal access_lock
        msg = "begin to parse variable declaration '{}'".format(input_text)
        log_enter_job_or_task_(parent_node, msg)
        #
        try:
            ttdeclaration = translator.parse_declaration(input_text)
            variables = translator.create_index_records_from_declaration(ttdeclaration)
        except Exception as e:
            utils.logging.log_exception(LOG_PREFIX,"_intrnl_parse_statements.ParseDeclarationTask_","failed: "+str(e))
            sys.exit(2)
        access_lock.acquire()
        parent_node._data["variables"] += variables
        access_lock.release()
        #
        msg = "parsed variable declaration '{}'".format(input_text)
        log_leave_job_or_task_(parent_node, msg)
    
    post_parsing_jobs = [] # jobs to run after the file was parsed statement by statement
    class ParseAttributesJob_:
        """
        :note: the term 'job' should highlight that an object of this class
        is put into a list that is submitted to a worker thread pool at the end of the parsing.
        """
        def __init__(self,parent_node,input_text):
            self._parent_node = parent_node
            self._input_text  = input_text
        def run(self):
            nonlocal access_lock
            msg = "begin to parse attributes statement '{}'".format(self._input_text)
            log_enter_job_or_task_(self._parent_node, msg)
            #
            attribute, modified_vars = \
                translator.parse_attributes(translator.attributes.parseString(self._input_text)[0])
            for var_context in self._parent_node._data["variables"]:
                if var_context["name"] in modified_vars:
                    access_lock.acquire()
                    var_context["qualifiers"].append(attribute)
                    access_lock.release()
            #
            msg = "parsed attributes statement '{}'".format(self._input_text)
            log_leave_job_or_task_(self._parent_node, msg)
    class ParseAccDeclareJob_:
        """
        :note: the term 'job' should highlight that an object of this class
        is put into a list that is submitted to a worker thread pool at the end of the parsing.
        """
        def __init__(self,parent_node,input_text):
            self._parent_node = parent_node
            self._input_text  = input_text
        def run(self):
            nonlocal access_lock
            msg = "begin to parse acc declare directive '{}'".format(self._input_text)
            log_enter_job_or_task_(self._parent_node, msg)
            #
            parse_result = translator.acc_declare.parseString(self._input_text)[0]
            for var_context in self._parent_node._data["variables"]:
                for var_name in parse_result.map_alloc_variables():
                    if var_context["name"] == var_name:
                        access_lock.acquire()
                        var_context["declare_on_target"] = "alloc"
                        access_lock.release()
                for var_name in parse_result.map_to_variables():
                    if var_context["name"] == var_name: 
                        access_lock.acquire()
                        var_context["declare_on_target"] = "to"
                        access_lock.release()
                for var_name in parse_result.map_from_variables():
                    if var_context["name"] == var_name: 
                        access_lock.acquire()
                        var_context["declare_on_target"] = "from"
                        access_lock.release()
                for var_name in parse_result.map_tofrom_variables():
                    if var_context["name"] == var_name: 
                        access_lock.acquire()
                        var_context["declare_on_target"] = "tofrom"
                        access_lock.release()
            msg = "parsed acc declare directive '{}'".format(self._input_text)
            log_leave_job_or_task_(self._parent_node, msg)

    # Parser events
    root        = Node("root","root",data=index,parent=None)
    current_node = root
    current_statement = None

    def create_base_entry_(kind,name,filepath):
        entry = {}
        entry["kind"]        = kind
        entry["name"]        = name
        #entry["file"]        = filepath
        entry["variables"]   = []
        entry["types"]       = []
        entry["subprograms"] = []
        entry["used_modules"] = []
        return entry
    def log_enter_node_():
        nonlocal current_node
        nonlocal current_statement
        utils.logging.log_debug(LOG_PREFIX,"_intrnl_parse_statements","[current-node={0}:{1}] enter {2} '{3}' in statement: '{4}'".format(\
          current_node._parent._kind,current_node._parent._name,
          current_node._kind,current_node._name,\
          current_statement))
    def log_leave_node_():
        nonlocal current_node
        nonlocal current_statement
        utils.logging.log_debug(LOG_PREFIX,"_intrnl_parse_statements","[current-node={0}:{1}] leave {0} '{1}' in statement: '{2}'".format(\
          current_node._data["kind"],current_node._data["name"],\
          current_statement))
    def log_detection_(kind):
        nonlocal current_node
        nonlocal current_statement
        utils.logging.log_debug2(LOG_PREFIX,"_intrnl_parse_statements","[current-node={}:{}] found {} in statement: '{}'".format(\
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
                    if q=="global" or "device" in q:
                        accelerator_routine = True
            if current_node._kind == "type" or accelerator_routine: 
                end = (current_linemap_no,current_statement_no)
                current_node._data["statements"] = current_node.statements(linemaps,end)
            current_node = current_node._parent
    def ModuleStart(tokens):
        nonlocal root
        nonlocal current_node
        name = tokens[0]
        module = create_base_entry_("module",name,filepath)
        assert current_node == root
        current_node._data.append(module)
        current_node = Node("module",name,data=module,parent=current_node)
        log_enter_node_()
    def ProgramStart(tokens):
        nonlocal root
        nonlocal current_node
        name    = tokens[0]
        program = create_base_entry_("program",name,filepath)
        assert current_node._kind == "root"
        current_node._data.append(program)
        current_node = Node("program",name,data=program,parent=current_node)
        log_enter_node_()
    #host|device,name,[args]
    def SubroutineStart(tokens):
        global LOG_PREFIX
        nonlocal current_statement
        nonlocal current_node
        log_detection_("start of subroutine")
        if current_node._kind in ["root","module","program","subroutine","function"]:
            name = tokens[1]
            subroutine = create_base_entry_("subroutine",name,filepath)
            subroutine["attributes"]      = [q.lower() for q in tokens[0]]
            subroutine["dummy_args"]       = list(tokens[2])
            if current_node._kind == "root":
                current_node._data.append(subroutine)
            else:
                current_node._data["subprograms"].append(subroutine)
            current_node = Node("subroutine",name,data=subroutine,parent=current_node)
            current_node._begin = (current_linemap_no,current_statement_no)
            log_enter_node_()
        else:
            utils.logging.log_warning(LOG_PREFIX,"_intrnl_parse_statements","found subroutine in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))
    #host|device,name,[args],result
    def FunctionStart(tokens):
        global LOG_PREFIX
        nonlocal current_statement
        nonlocal current_node
        log_detection_("start of function")
        if current_node._kind in ["root","module","program","subroutine","function"]:
            name = tokens[1]
            function = create_base_entry_("function",name,filepath)
            function["attributes"]  = [q.lower() for q in tokens[0]]
            function["dummy_args"]  = list(tokens[2])
            function["result_name"] = name if tokens[3] is None else tokens[3]
            if current_node._kind == "root":
                current_node._data.append(function)
            else:
                current_node._data["subprograms"].append(function)
            current_node = Node("function",name,data=function,parent=current_node)
            current_node._begin = (current_linemap_no,current_statement_no)
            log_enter_node_()
        else:
            utils.logging.log_warning(LOG_PREFIX,"_intrnl_parse_statements","found function in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(current_statement,current_node._kind))
    def TypeStart(tokens):
        global LOG_PREFIX
        nonlocal current_linemap_no
        nonlocal current_statement
        nonlocal current_statement_no
        nonlocal current_node
        log_detection_("start of type")
        if current_node._kind in ["module","program","subroutine","function"]:
            assert len(tokens) == 2
            name = tokens[1]
            derived_type = {}
            derived_type["name"]      = name
            derived_type["kind"]      = "type"
            derived_type["variables"] = []
            derived_type["types"]     = []
            current_node._data["types"].append(derived_type)
            current_node = Node("type",name,data=derived_type,parent=current_node)
            current_node._begin = (current_linemap_no,current_statement_no)
            log_enter_node_()
        else:
            utils.logging.log_warning(LOG_PREFIX,"_intrnl_parse_statements","found derived type in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
                    format(current_statement,current_node._kind))
    def Use(tokens):
        nonlocal current_node
        log_detection_("use statement")
        if current_node._kind != "root":
            used_module = {}
            used_module["name"] = translator.make_f_str(tokens[1])
            used_module["only"] = []
            for pair in tokens[2]:
                original = translator.make_f_str(pair[0])
                renamed = original if pair[1] is None else translator.make_f_str(pair[1])
                used_module["only"].append({ "original": original, "renamed": renamed })
            current_node._data["used_modules"].append(used_module) # TODO only include what is necessary
    
    # delayed parsing
    
    def Declaration(tokens):
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        nonlocal task_executor
        nonlocal total_num_tasks
        #print(current_statement)
        log_detection_("declaration")
        if current_node != root:
            total_num_tasks += 1
            task_executor.submit(ParseDeclarationTask_,current_node,current_statement) 
    def Attributes(tokens):
        """
        Add attributes to previously declared variables in same scope/declaration list.
        Does not modify scope of other variables.
        """
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        #print(current_statement)
        log_detection_("attributes statement")
        if current_node != root:
            job = ParseAttributesJob_(current_node,current_statement) 
            post_parsing_jobs.append(job)
    def AccDeclare():
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        log_detection_("acc declare directive")
        if current_node != root:
            job = ParseAccDeclareJob_(current_node,current_statement) 
            post_parsing_jobs.append(job)
    
    def AccRoutine():
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current_node
        nonlocal current_statement
        log_detection_("acc routine directive")
        if current_node != root:
            parse_result = translator.acc_routine.parseString(current_statement)[0]
            if parse_result.parallelism() == "seq":
                current_node._data["attributes"] += ["host","device"]
            elif parse_result.parallelism() == "gang":
                current_node._data["attributes"] += ["host","device:gang"]
            elif parse_result.parallelism() == "worker":
                current_node._data["attributes"] += ["host","device:worker"]
            elif parse_result.parallelism() == "vector":
                current_node._data["attributes"] += ["host","device:vector"]

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

    def try_to_parse_string(expression_name,expression):
        try:
           expression.parseString(current_statement)
           return True
        except ParseBaseException as e: 
           utils.logging.log_debug3(LOG_PREFIX,"_intrnl_parse_statements","did not find expression '{}' in statement '{}'".format(expression_name,current_statement))
           utils.logging.log_debug4(LOG_PREFIX,"_intrnl_parse_statements",str(e))
           return False
    def is_end_statement_(tokens,kind):
        result = tokens[0] == "end"+kind
        if not result and len(tokens):
            result = tokens[0] == "end" and tokens[1] == kind
        return result
    def consider_statement(stripped_statement):
        passes_filter = p_filter.match(stripped_statement) != None
        return passes_filter

    # parser loop
    for current_linemap_no, current_linemap in enumerate(linemaps):
        if current_linemap["is_active"]:
            for current_statement_no,stmt in enumerate(current_linemap["statements"]):
                current_statement = stmt["body"].lower().strip(" \t\n")
                if not consider_statement(current_statement):
                    utils.logging.log_debug3(LOG_PREFIX,"_intrnl_collect_statements","ignore statement '{}'".format(current_statement))
                else:
                    utils.logging.log_debug3(LOG_PREFIX,"_intrnl_parse_statements","process statement '{}'".format(current_statement))
                    current_tokens              = re.split(r"\s+|\t+",current_statement.lower().strip(" \t"))
                    current_statement_stripped  = "".join(current_tokens)
                    for expr in ["program","module","subroutine","function","type"]:
                        if is_end_statement_(current_tokens,expr):
                             End()
                    for comment_char in "!*c":
                        if current_tokens[0] == comment_char+"$acc":
                            if current_tokens[1] == "declare":
                                AccDeclare()
                            elif current_tokens[1] == "routine":
                                AccRoutine()
                    if current_tokens[0] == "use":
                        try_to_parse_string("use",use)
                    #elif current_tokens[0] == "implicit":
                    #    try_to_parse_string("implicit",IMPLICIT)
                    elif current_tokens[0] == "module":
                        try_to_parse_string("module",module_start)
                    elif current_tokens[0] == "program":
                        try_to_parse_string("program",program_start)
                    elif current_tokens[0].startswith("type"): # type a ; type, bind(c) :: a
                        try_to_parse_string("type",type_start)
                    elif current_tokens[0].startswith("attributes"): # attributes(device) :: a
                        try_to_parse_string("attributes",attributes)
                    # cannot be combined with above checks
                    if "function" in current_tokens:
                        try_to_parse_string("function",function_start)
                    elif "subroutine" in current_tokens:
                        try_to_parse_string("subroutine",subroutine_start)
                    for expr in ["type","character","integer","logical","real","complex","double"]: # type(dim3) :: a 
                       if expr in current_tokens[0]:
                           try_to_parse_string("declaration",datatype_reg)
                           break
                    #try_to_parse_string("declaration|type_start|use|attributes|module_start|program_start|function_start|subroutine_start",\
                    #  datatype_reg|type_start|use|attributes|module_start|program_start|function_start|subroutine_start)
    task_executor.shutdown(wait=True) # waits till all tasks have been completed

    # apply attributes and acc variable modifications
    num_post_parsing_jobs = len(post_parsing_jobs)
    if num_post_parsing_jobs > 0:
        utils.logging.log_debug(LOG_PREFIX,"_intrnl_parse_statements","apply variable modifications (submit {} jobs to worker pool of size {})".format(\
          num_post_parsing_jobs,PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE))
        with concurrent.futures.ThreadPoolExecutor(\
            max_workers=PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE)\
                as job_executor:
            for job in post_parsing_jobs:
                job_executor.submit(job.run)
        utils.logging.log_debug(LOG_PREFIX,"_intrnl_parse_statements","apply variable modifications --- done") 
        post_parsing_jobs.clear()

    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_parse_statements") 
    return index

def _intrnl_write_json_file(index,filepath):
    global PRETTY_PRINT_INDEX_FILE
    global LOG_PREFIX    
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_write_json_file",{"filepath":filepath}) 
    
    with open(filepath,"wb") as outfile:
         if PRETTY_PRINT_INDEX_FILE:
             outfile.write(orjson.dumps(index,option=orjson.OPT_INDENT_2))
         else:
             outfile.write(orjson.dumps(index))
    
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_write_json_file") 

def _intrnl_read_json_file(filepath):
    global LOG_PREFIX    
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_read_json_file",{"filepath":filepath}) 
    
    with open(filepath,"rb") as infile:
         utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_read_json_file") 
         return orjson.loads(infile.read())

# API
def scan_file(filepath,preproc_options,index):
    """
    Creates an index from a single file.
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"scan_file",{"filepath":filepath,"preproc_options":preproc_options}) 
    
    filtered_statements = _intrnl_read_fortran_file(filepath,preproc_options)
    utils.logging.log_debug2(LOG_PREFIX,"scan_file","extracted the following statements:\n>>>\n{}\n<<<".format(\
        "\n".join(filtered_statements)))
    index += _intrnl_parse_statements(filtered_statements,filepath)
    
    utils.logging.log_leave_function(LOG_PREFIX,"scan_file") 

def update_index_from_linemaps(linemaps,index):
    """Updates index from a number of linemaps."""
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"update_index_from_linemaps") 
    
    if len(linemaps):
        index += _intrnl_parse_statements(linemaps,filepath=linemaps[0]["file"])
    
    utils.logging.log_leave_function(LOG_PREFIX,"update_index_from_linemaps") 

def write_gpufort_module_files(index,output_dir):
    """
    Per module / program found in the index
    write a GPUFORT module file.
    
    :param list index:    [in] Empty or non-empty list.
    :param str output_dir: [in] Output directory.
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"write_gpufort_module_files",{"output_dir":output_dir})
    
    for mod in index:
        filepath = output_dir + "/" + mod["name"] + GPUFORT_MODULE_FILE_SUFFIX
        _intrnl_write_json_file(mod,filepath)
    
    utils.logging.log_leave_function(LOG_PREFIX,"write_gpufort_module_files")

def load_gpufort_module_files(input_dirs,index):
    """
    Load gpufort module files and append to the index.

    :param list input_dirs: [in] List of input directories (as strings).
    :param list index:     [inout] Empty or non-empty list. Loaded data structure is appended.
    """
    global LOG_PREFIX
    utils.logging.log_enter_function(LOG_PREFIX,"load_gpufort_module_files",{"input_dirs":",".join(input_dirs)})
    
    for input_dir in input_dirs:
         for child in os.listdir(input_dir):
             if child.endswith(GPUFORT_MODULE_FILE_SUFFIX):
                 module_already_exists = False
                 for mod in index:
                     if mod == child.replace(GPUFORT_MODULE_FILE_SUFFIX,""):
                         module_already_exists = True
                         break
                 if not module_already_exists:
                     mod_index = _intrnl_read_json_file(os.path.join(input_dir, child))
                     index.append(mod_index)
    
    utils.logging.log_leave_function(LOG_PREFIX,"load_gpufort_module_files")