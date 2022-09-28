# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import opts
from .. import analysis
from .. import tree
from .. import optvals

from . import loops
from . import acctypes

loops.single_level_indent = opts.single_level_indent

def _check_loop_parallelism(resource_filter,acc_loop_info)
    if acc_loop_info.gang.specified:
        if resource_filter.gang_partitioned_mode():
            raise util.eror.SyntaxError("already in gang-partitioned region")
        elif resource_filter.worker_partitioned_mode():
            raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
        elif resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
    if acc_loop_info.worker.specified:
        if resource_filter.worker_partitioned_mode():
            raise util.eror.SyntaxError("already in worker-partitioned region")
        elif resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
    if acc_loop_info.vector.specified:
        if resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("already in vector-partitioned region")
        
def _create_acc_resource_filter(
    initially_gang_partitioned = False,
    initially_worker_partitioned = False,
    initially_vector_partitioned = False):
  """Initialize gang, worker, vector resources with
  None for all initially partitioned resource types.
  This indicates that all available resources can be used
  for that resource type. In particular, this is
  used when translating OpenACC routines.
  """
  return loops.AccResourceFilter(
    num_gangs = [None] if initially_gang_partitioned else [],
    num_workers = [None] if initially_worker_partitioned else [],
    vector_length = [None] if initially_vector_partitioned else []
  )

def _create_unpartitioned_loop(ttdo):
    return loops.Loop(
      index = ttnode.loop_var(),
      first = ttnode.begin_cstr(),
      last = ttnode.end_cstr(),
      step = ttnode.step_cstr() if ttnode.has_step() else None
    )

def _render_loopnest(loopnest_info)
    loopnest = loopnest_info.loopnest
    private_vars = loopnest_info.private_vars
    reductions = loopnest_info.reductions
    num_collapse = loopnest_info.num_collapse
    tile_sizes = loopnest_info.tile_sizes
    return render.render_loopnest(
      loopnest,
      private_vars,
      reductions,
      num_collapse,
      tile_sizes
    )

def transform_acc_construct(
      ttaccconstruct,
      scope,
      device_type, 
      initially_gang_partitioned = False,
      initially_worker_partitioned = False,
      initially_vector_partitioned = False):
    """Transform an OpenACC compute construct or routine body to HIP C++.
    :param scope: A scope object, see GPUFORT's indexer component.
    :param str device_type: The device type (`acc_device_nvidia`, `acc_device_radeon`).
    :param bool initially_gang_partitioned: Start in gang-partitioned mode.
    :param bool initially_worker_partitioned: Start in worker-partitioned mode.
    :param bool initially_vector_partitioned: Start in vector-partitioned mode.
    """
    trafo_result = TransformationResult()
    # for the whole compute construct
    trafo_result.generated_code = ""
    resource_filter = _create_acc_resource_filter(
      initially_gang_partitioned,
      initially_worker_partitioned,
      initially_vector_partitioned)
    indent = ""
    # per loopnest in the compute construct
    loopnest_info = None
    def traverse_container_body_(ttcontainer):
        nonlocal trafo_result
        nonlocal indent
        nonlocal resource_filter
        previous_indent = indent
        indent += ttcontainer.indent
        previous_indent2 = indent

        statement_selector_is_open = False
        num_children = len(ttcontainer)
        for i,child in enumerate(ttcontainer):
            if isinstance(child,tree.TTContainer):
                if statement_selector_is_open:
                    indent = previous_indent2
                    trafo_result.generated_code += indent+"}\n"
                    traverse_(child)
            else:
                if ( not statement_selector_is_open
                     and not resource_filter.worker_and_vector_partitioned_mode()
                     and not i == num_children-1 ):
                    trafo_result.generated_code += "{}if ({}) {{".format(
                      indent,
                      resource_filter.statement_selection_condition()
                    )
                    statement_selector_is_open = True
                    indent += opts.single_level_indent
                 traverse_(child)
        if statement_selector_is_open:
            trafo_result.generated_code += indent+"}\n"
        indent = previous_indent

    def traverse_(ttnode):
        nonlocal trafo_result
        nonlocal indent
        nonlocal resource_filter
        nonlocal loopnest_info
    
        lvalues, rvalues = [], []
        if isinstance(ttnode,IComputeConstruct):
            acc_construct_info = analysis.acc.analyze_directive(ttnode)  
            if acc_construct_info.is_serial:
                trafo_result.max_num_gangs = "1"    
                trafo_result.max_num_workers = "1"    
                trafo_result.max_vector_length = "1"    
            else:
                if acc_construct_info.num_gangs.specified:
                    trafo_result.max_num_gangs = acc_construct_info.num_gangs
                if acc_construct_info.num_workers.specified:
                    trafo_result.max_num_workers = acc_construct_info.num_workers
                if acc_construct_info.vector_length.specified:
                    trafo_result.max_vector_length = acc_construct_info.vector_length
            if acc_construct_info.private_variables.specified: 
                trafo_result.private_variables = acc_construct_info.private_variables
                trafo_result.generated_code += render.render_private_variables_decl_list(
                  ttvalues,
                  scope
                )
            if acc_construct_info.firstprivate_variables.specified: 
                trafo_result.firstprivate_variables = acc_construct_info.firstprivate_variables

        elif isinstance(ttnode,TTDo):
            loopnest_open = ""
            loopnest_close = ""
            loopnest_resource_filter = loops.AccResourceFilter()
            loopnest_indent = ""
            #
            if loopnest_info != None
                loopnest = loopnest_info.loopnest
                num_collapse = loopnest_info.num_collapse 
                tile_sizes = loopnest_info.tile_sizes
            else:
                loopnest = None
                num_collapse = 0
                tile_sizes = []
            if num_collapse > 1:
                assert loopnest != none
                loopnest.append(
                  _create_unpartitioned_loop(ttdo)
                )      
                if len(loopnest) == num_collapse:
                    loopnest_open,\
                    loopnest_close,\
                    loopnest_resource_filter,\
                    loopnest_indent = \
                        _render_loopnest(loopnest_info)
                    trafo_result.append(loopnest_info)
                    loopnest_info = None
                    num_collapse = 0   
                    # TODO check if we can move statement into inner loop, should be possible if 
                    # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
                    # alternative interpretation of collapse user information -> we can reorder statements without error
            elif len(tile_sizes) > 0:
                assert loopnest != None
                loopnest.append(
                  _create_unpartitioned_loop(ttdo)
                )      
                if len(loopnest) == len(tile_sizes):
                    loopnest_open,\
                    loopnest_close,\
                    loopnest_resource_filter,\
                    loopnest_indent = \
                      _render_loopnest(loopnest)
                    trafo_result.append(loopnest_info)
                    loopnest_info = None
                    tile_sizes = []
                traverse_container_body_(ttnode)
            elif ttnode.annotation != None:
                loop_annotation = ttnode.annotation
                acc_loop_info = analysis.analyze_directive(ttnode,device_type) 
                _check_loop_parallelism(resource_filter,loop_parallelism)
                loopnest_info.private_vars = loop_annotation.private_vars() 
                loopnest = loops.Loopnest([
                  loops.Loop(
                    index = ttnode.loop_var(),
                    first = ttnode.begin_cstr(),
                    last = ttnode.end_cstr(),
                    step = ttnode.step_cstr() if ttnode.has_step() else None,
                    num_gangs = acc_loop_info.gang.value,
                    num_workers = acc_loop_info.worker.value,
                    vector_length = acc_loop_info.vector.value,
                    gang_partitioned = acc_loop_info.gang.specified,
                    worker_partitioned = acc_loop_info.worker.specified,
                    vector_partitioned = acc_loop_info.vector.specified
                  )
                ])
                if ( num_collapse == 0
                     and not len(tile_sizes) ):
                    loopnest_open,\
                    loopnest_close,\
                    loopnest_resource_filter,\
                    loopnest_indent = \
                      _render_loopnest(loopnest)
                    trafo_result.append(loopnest_info)
                    loopnest_info = None
                # loop directives might contain lvalues, rvalues that need to be passed
                _find_lvalues_and_rvalues_in_directive(ttnode,lvalues,rvalues)
            else:
                pass   
            resource_filter += loopnest_resource_filter
            previous_indent = indent
            indent += loopnest_indent
            #
            traverse_container_body_(ttnode)
            #
            resource_filter -= loopnest_resource_filter
            indent = previous_indent
        elif isinstance(ttnode,TTContainer):
            _find_lvalues_and_rvalues(ttnode,lvalues,rvalues)
            # TODO scan lvalue and rvalue
            trafo_result.generated_code += textwrap.indent(ttnode.header_cstr(),indent)
            indent += ttnode.indent
            #
            traverse_container_body_(ttnode)
            #
            trafo_result.generated_code += ttnode.footer_cstr()
        else: # other statements
            _find_lvalues_and_rvalues(ttnode,lvalues,rvalues)
            # TODO scan for lvalue and rvalues
            # TODO expand array assignment expressions
            # TODO modify expressions that are
            trafo_result.generated_code += ttnode.cstr().rstrip("\n")+"\n"
    traverse_(ttaccconstruct)
    return trafo_result
