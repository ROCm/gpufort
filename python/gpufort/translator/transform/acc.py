# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""

Concepts:

* Private variables are implemented as shown in the snippet below,
  where the prefix $ indicates the lines that are injected 
  by transformations.
  
  ``` 
  declare var
  parallel for i = 1,N
     $declare var # hides parent var
     ... do usual stuff with var
  end loop
  ```

  The transformation simply inserts a local variable,
  into the loop that hides the parent scope's variable.

* Firstprivate variables are implemented as shown in the snippet below,
  where the prefix $ indicates the lines that are injected 
  by transformations.
  

  ``` 
  declare var
  $declare parent_var_copy = var
  parallel for i = 1,N
     $declare var # hides parent var
     $var = parent_var_copy # copy initial value from parent var's copy
     ... do usual stuff with var
  end loop
  ```

  As for the private variables, the transformation inserts a local variable,
  into the loop that hides the parent scope's variable.

* Reductions:

  * Global reductions:

    Gang reductions are implemented as shown

  * Blockwise reductions:
 
    Blockwise reductions are implemented as shown in the snippet below,
    where the prefix $ indicates the lines that are injected 
    by transformations.

    ``` 
    declare var                                   # per gang, might need to be reduced further
    $declare var_buffer[N]                        # N is the size of the threadblock
    parallel for i = 1,N                          # -> mapped to threads
       $var_buffer[i] = var
       $for j = 1, M                              # M is the work per thread
          $declare var                            # hides parent var
          $var = <init_val>
         ... do usual stuff with var
         $var_buffer[i] = reduce(var_buffer[i],var)
       $end loop
    end loop
    $for i = 1,N
       $var = reduce(var_buffer[i],var)    
    $end for
  ```
  
  The transformation needs to insert a buffer allocation before the parellel loop,
  a local variable declaration within the loop, a buffer entry update at
  the end of the loop, and an aggregation loop after the loop, which must be performed
  by the current gang or worker leader.
"""
from gpufort import util

from .. import tree
from .. import conv
from . import loops
from . import loops2

class AccParallelismMode:

    """Class for encoding OpenACC parallelism level."""

    def __init__(self):
        self._gang = None
        self._worker = None
        self._vector = None
        self._gang_specified = False
        self._worker_specified = False
        self._vector_specified = False

    @property
    def gang(self):
        return self._gang

    @property
    def worker(self):
        return self._worker

    @property
    def vector(self):
        return self._vector

    @gang.setter
    def gang(self, gang):
        assert not self._gang_specified
        self._gang_specified = True
        self._gang = gang

    @worker.setter
    def worker(self, worker):
        assert not self._worker_specified
        self._worker_specified = True
        self._worker = worker

    @vector.setter
    def vector(self, vector):
        assert not self._vector_specified
        self._vector_specified = True
        self._vector = vector

    @property
    def gang_specified(self):
        return self._gang_specified

    @property
    def worker_specified(self):
        return self._worker_specified

    @property
    def vector_specified(self):
        return self._vector_specified

    @property
    def can_specify_gang(self):
        return not (
            self._gang_specified
            or self._worker_specified
            or self._vector_specified
        )

    @property
    def can_specify_worker(self):
        return not (
            self._worker_specified
            or self._vector_specified
        )

    @property
    def can_specify_vector(self):
        return not (
            self._vector_specified
        )

    @property
    def constraints(self):
        """Generates a list of constraints for masking assignment-like statements
        depending on the current parallelism specification.
        """
        constraints = []
        if self._gang_specified and self._gang != None:
            constraints.append("hipBlockIdx.x < {}".format(
                self._gang if self._gang != None else self.max_num_gangs
            ))
        if self._worker_specified and self._worker != None:
            constraints.append("hipThreadIdx.y < {}".format(self._worker))
        elif not self._worker_specified:
            constraints.append("hipThreadIdx.y == 0".format(self._worker))
        if self._vector_specified and self._vector != None:
            constraints.append("hipThreadIdx.x < {}".format(self._vector))
        elif not self._vector_specified:
            constraints.append("hipThreadIdx.x == 0".format(self._vector))
        return constraints

    def unset_gang(self):
        self._gang_specified = False
        self._gang = None

    def unset_worker(self):
        self._worker_specified = False
        self._worker = None

    def unset_vector(self):
        self._vector_specified = False
        self._vector = None

class ACC2HIPTransformer(loops2.HipGenerator):

    def __init__(self, device_type):
        self.device_type = device_type
        # state
        self._parallelism_mode = AccParallelismMode()
        self._statement_filter_node = None
        

    def transform(self, container: tree.TTContainer):
        """Recursively unrolls all OpenACC constructs, found in the container, descends into every
        container statement in the body.
        :param str device_type: The device type identifier for
        which to perform the transformation.
        :note: Assumes that semantics checks have been performed.
        :note: Propagates (assignment, function call) statement filter down from the top
            of the tree to the leaves
        """
        # TODO augment containers on with artificial nodes on the way down
        # TODO transform containers to different nodes on the way up
        newbody = []  # shallow copy
        # todo reduce resources here too
        last = -1
        for pos, ttstmt in enumerate(container.body):
            # pre-process
            if isinstance(ttstmt, (tree.TTAccParallelLoop, tree.TTAccKernelsLoop)):
                new_stmt = self._split_combined_construct(container, pos)
                container.body[pos] = new_stmt
            elif isinstance(ttstmt, tree.TTAccLoop):
                self._touch_compute_construct(ttstmt)
            elif isinstance(ttstmt, tree.TTAccLoop):
                self._touch_loop_directive_first(ttstmt)
            else:
                self._touch_statement(ttstmt)
            # descend
            if isinstance(ttstmt, tree.TTContainer):
                self.transform(ttstmt)
            # ascend, post process
            if isinstance(ttstmt, tree.TTAccLoop):
                self._touch_loop_directive_last(ttstmt)

            # if isinstance(ttstmt,tree.TTAccSerial):
            # resource limitations
            # no reduction
            # subst = ttstmt
            # newbody[pos] = subst
            # if isinstance(ttstmt,tree.TTAccParallel):
            # resource limitations
            # no reduction
            # subst = ttstmt
            # newbody[pos] = subst
            # elif isinstance(ttstmt,tree.TTAccKernels):
            # resource limitations
            # no reduction
            # subst = ttstmt
            # newbody[pos] = subst
            # elif isinstance(ttstmt,tree.TTAccLoop):
            # resource limitations
            # update filter
            # subst = tree.TTSubstLoopDirective(ttstmt)
            # subst.body = list(orig.body) # shallow copy
            # newbody[pos] = subst
        #
        # ttcontainer.body = newbody

    def _split_combined_construct(self, ttstmt_orig):
        """Splits a combined OpenACC construct (`acc kernels loop`, `acc parallel loop`)
        into its compute construct part and acc loop parts.
        Replaces the orginal node by a substitution container
        that stores the original object in a field and the new subtree
        in its body.
        """
        assert type(ttstmt_orig) in [
            tree.TTAccKernelsLoop, tree.TTAccParallelLoop]
        if isinstance(ttstmt_orig, tree.TTAccKernelsLoop):
            new_acc_construct = ttstmt_orig.acc_kernels()
        else:
            new_acc_construct = ttstmt_orig.acc_parallel()
        new_acc_loop = ttstmt_orig.acc_loop()
        new_acc_construct.append(new_acc_loop)
        new_acc_loop.body = list(ttstmt_orig.body)
        ttstmt_new = tree.TTSubstContainer(ttstmt_orig)
        ttstmt_new.append(new_acc_construct)
        return ttstmt_new

    def _touch_compute_construct(self, ttaccdir):
        """Inject private and firstprivate variables for all constructs. 
        Fix gang, worker, and vector to 1 for tree.TTAccSerial.
        """
        if isinstance(ttaccdir, tree.TTAccSerial):
            self._parallelism_mode.gang = tree.TTNumber(["1"])
            self._parallelism_mode.worker = tree.TTNumber(["1"])
            self._parallelism_mode.vector = tree.TTNumber(["1"])
        for clause in ttaccdir.walk_clauses_device_type(self.device_type):
            if isinstance(clause, tree.TTAccClausePrivate):
                self._inject_private_var_decls(ttaccdir,clause.var_list)
            elif isinstance(clause, tree.TTAccClauseFirstprivate):
                self._inject_firstprivate_var_decls(ttaccdir,clause.var_list)

    def _touch_loop_directive_first(self, ttaccdir: tree.TTAccLoop):
        """Set parallelism mode according to the loop directive's clauses.
        """
        for clause in ttaccdir.walk_clauses_device_type(self.device_type):
            if isinstance(clause, tree.TTAccClauseGang):
                if not self._parallelism_mode.can_specify_gang:
                    raise util.error.TransformationError(
                        "already in gang-partitioned mode")
                self._parallelism_mode.gang = clause.arg
            elif isinstance(clause, tree.TTAccClauseWorker):
                if not self._parallelism_mode.can_specify_worker:
                    raise util.error.TransformationError(
                        "already in worker-partitioned mode")
                self._parallelism_mode.worker = clause.arg
            elif isinstance(clause, tree.TTAccClauseVector):
                if not self._parallelism_mode.can_specify_vector:
                    raise util.error.TransformationError(
                        "already in vector-partitioned mode")
                self._parallelism_mode.vector = clause.arg

    def _touch_loop_directive_last(self, ttaccdir: tree.TTAccLoop):
        """Unset the parallelism mode that the loop directive has specified.
        """
        for clause in ttaccdir.walk_clauses_device_type(self.device_type):
            if isinstance(clause, tree.TTAccClauseGang):
                self._parallelism_mode.unset_gang()
            elif isinstance(clause, tree.TTAccClauseWorker):
                self._parallelism_mode.unset_worker()
            elif isinstance(clause, tree.TTAccClauseVector):
                self._parallelism_mode.unset_vector()

    def _touch_statement(self, ttcontainer, ttstmt):
        """Append a statement to the body of the statement filter node.
        """
        if self._is_subject_to_statement_filter(ttstmt):
            statement_filter = " && ".join(self._parallelism_mode.constraints)
            if self._statement_filter_node == None:
                self._statement_filter_node = tree.TTCElseIf(statement_filter)
            self._statement_filter_node.body.append(ttstmt)
        else:
            if self._statement_filter_node != None:
                ttcontainer.append(self._statement_filter_node)
                self._statement_filter_node = None
            ttcontainer.append(ttstmt)

    def _is_subject_to_statement_filter(self, ttstmt):
        return isinstance(ttstmt, (
            tree.TTAssignment,
            tree.TTUnrolledArrayAssignment,
            tree.TTSubroutineCall,
        ))

    def _inject_private_var_decls(self, ttcontainer, var_list):
        """:note: Assumes semantics check has been performed on all variables.
        """
        nodes_to_prepend = []
        for ttvalue in var_list:
            nodes_to_prepend += self._derive_private_decl_nodes(
                ttvalue.name, ttvalue.symbol_info
            )
        for node in reversed(nodes_to_prepend):
            ttcontainer.body.insert(0, node)

    def _inject_firstprivate_var_decls(self, ttcontainer, var_list):
        """:note: Assumes semantics check has been performed on all variables.
        """
        nodes_to_prepend = []
        for ttvalue in var_list:
            nodes_to_prepend += self._derive_private_decl_nodes(
                ttvalue.name, ttvalue.symbol_info
            )
            src_name = self._label_firstprivate_argument(ttvalue.name),
            if ttvalue.rank > 0:
                rw_index = self.label_generator("idx")
                nodes_to_prepend.append(
                    tree.TTCCopyForLoop(  # dest,src,dest_idx,src_idx,n
                        ttvalue.name,
                        src_name,
                        rw_index,
                        rw_index,
                        ttvalue.symbol_info.size_expr.cstr()
                    )
                )
            else:
                nodes_to_prepend.append(
                    tree.TTCCopyStatement(ttvalue.name, src_name)
                )
        for node in reversed(nodes_to_prepend):
            ttcontainer.body.insert(0, node)

    def _inject_reduction_var_decl_and_reduction_in_construct(self, ttloop, op, var_list):
        """
        """
        nodes_to_prepend = []
        for ttvalue in var_list:
            nodes_to_prepend += self._derive_private_decl_nodes(
                ttvalue.name, ttvalue.symbol_info
            )
            result_name = self._label_reduction_argument(ttvalue.name)
            c_type = conv.c_type(
                self.symbol_info.type,
                self.bytes_per_element
            )
            c_init_val = conv.reduction_c_init_val(self.op, c_type)
            if ttvalue.rank > 0:
                nodes_to_prepend.append(
                    tree.TTCCopyForLoop(  # dest,src,idx,n
                        ttvalue.name,
                        c_init_val,
                        self.label_generator("idx"),
                        None,
                        ttvalue.symbol_info.size_expr.cstr()
                    )
                )
            else:
                nodes_to_prepend.append(
                    tree.TTCCopyStatement(ttvalue.name, c_init_val)
                )
        for node in reversed(nodes_to_prepend):
            ttloop.body.insert(0, node)

    def _inject_reduction_buffer_decl_and_aggregation_in_construct_parent(self,
                                                                          ttconstructparent,
                                                                          ttconstruct,
                                                                          block_size,
                                                                          op,
                                                                          var_list
                                                                          ):
        """
        :param ttconstructparent: Container that contains the construct
                            that performs the reduction.
        :param ttconstruct: acc loop that performs the reduction. 
        :note: Assumes semantics check has been performed on all variables.
        """
        # append first and then prepend to not change the position of the loop
        nodes_to_prepend = []
        nodes_to_append = []
        for ttvalue in var_list:
            c_type = conv.c_type(
                self.symbol_info.type,
                self.bytes_per_element
            )
            c_init_val = conv.reduction_c_init_val(self.op, c_type)
        #
        loop_pos = ttconstructparent.index(ttconstruct)
        for node in reversed(nodes_to_append):
            ttconstructparent.body.insert(loop_pos+1, node)
        for node in reversed(nodes_to_prepend):
            ttconstructparent.body.insert(loop_pos, node)

    def _derive_private_decl_nodes(self, var_name, symbol_info):
        if symbol_info.rank > 0:
            buffer_name = self._label_array_buffer(var_name)
            return [
                tree.TTCVarDeclFromFortranSymbol(buffer_name, symbol_info),
                tree.TTCGpufortArrayPtrDecl(var_name, symbol_info),
                tree.TTCGpufortArrayPtrWrap(
                    var_name, buffer_name, symbol_info),
            ]
        else:
            return [
                tree.TTCVarDeclFromFortranSymbol(var_name, symbol_info),
            ]

    def _label_array_buffer(self, name):
        return "_"+name+"_buffer"

    def _label_firstprivate_argument(self, name):
        return "_"+name+"_at_init"

    def _label_reduction_argument(self, name):
        return "_"+name+"_reduced"