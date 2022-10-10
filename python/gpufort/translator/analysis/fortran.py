# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import enum

from .. import tree

def _visit_values(expr,parents,lvalues,rvalues):
  """Traversal action that collects lvalues and rvalues, 
  excludes numbers and characters."""
  if isinstance(expr,tree.TTValue):
      if isinstance(expr._value,(tree.TTNumber,
                                 tree.TTCharacter)):
          pass
      elif isinstance(expr,tree.TTRvalue):
          rvalues.append(expr)
      elif isinstance(expr,tree.TTLvalue):
          lvalues.append(expr)

def find_lvalues_and_rvalues(ttnode,lvalues,rvalues):
    """Collect lvalues and rvalues, exclude numbers and characters."""
    tree.traversals.traverse(
        ttnode,
        _visit_values,
        tree.traversals.no_action,
        tree.traversals.no_crit,
        lvalues,
        rvalues)

def search_index_for_value(ttvalue,scope):
    pass

def search_scope_for_value(ttvalue,scope):
    pass

#class AssignmentType(enum.Enum):
#    UNIDENTIFIED = 0
#    LHS_SCALAR_RHS_SCALAR = 1
#    LHS_ARRAY_RHS_ARRAY = 2
#    LHS_SCALAR_RHS_REDUCTION_INTRINSIC_EVAL = 3
#
#class ArithExprInfo:
#    """:todo: For the time being, we ignore
#    all operations that transform the rank of a subexpression."""
#    def __init__(self,ttarithexpr):
#        self._ttarithexpr = ttarithexpr
#        self._rank = None
#        self._top_level_op = None
#        self._top_level_rvals = []
#            
#    def _traverse_main_rvals_and_ops(self,ttnode,action,depth=0):
#        if not action(self,ttnode,depth):
#            return
#        if isinstance(ttnode,pyparsing.ParseResults):
#            for child in ttnode._expr:
#                self._traverse_main_rvals_and_ops(child,action,depth+1)                  
#
#    def _compute_rank_action(self,ttnode,depth):
#        if isinstance(tree.TTRvalue):
#            if isinstance(tree._value,TTIdentifier):
#                       
#        return True
#             
#    def _get_top_level_op(self,ttnode,depth):
#        if isinstance(tree.TTRvalue):
#            pass
#        return depth == 0
#  
#    @property
#    def rank:
#        if self._rank == None:
#            self._traverse_main_rvals_and_ops(
#                self._ttarithexpr,
#                self._compute_rank_action)
#        return self._rank
            

#class AssignmentInfo:
#
#    def __init__(self,ttassignment):
#        self._type = AssignmentType.UNIDENTIFIED
#        self._assignment = ttassignment
#        self._main_rvals_and_ops = []
#        self._slices = []
#    @property
#    def lvalue(self):
#        return self._assignment._lhs
#    
#    def main_rvals_and_ops(self):
#        """The main rvalues and operators, i.e. does not include
#        rvalues and operators that appear in argument lists
#        of the rvalues of the assignment.
#        """
#        if not len(self._main_rvals_and_ops):
#            ttarithexpr = self._assignment._rhs._expr
#            def traverse_(ttnode):
#                if isinstance(ttnode,(tree.TTOperator,tree.TTRvalue)):
#                    self._main_rvals_and_ops.append(ttnode)
#                elif isinstance(ttnode,pyparsing.ParseResults):
#                    for child in ttnode._expr:
#                        if isinstance(child,(tree.TTOperator,tree.TTRvalue)):
#                            self._main_rvals_and_ops.append(child)
#                        elif isinstance(child,pyparsing.ParseResults)
#                            traverse_(child)                  
#        return self._main_rvals_and_ops
#    def _collect_slices(function_call_args,include_none_values=False):
#        ttranges = []
#        for i,ttnode in enumerate(function_call_args):
#            if isinstance(ttnode, tree.TTSlice):
#                ttranges.append(ttnode)
#            elif include_none_values:
#                ttranges.append(None)
#        return ttranges
#    
#    def _collect_slices_in_ttvalue(ttvalue,include_none_values=False):
#        """
#        :return A list of range objects. If the list is empty, no function call
#                or tensor access has been found. If the list contains a None element
#                this implies that a function call or tensor access has been found 
#                but a scalar index argument was used.
#        """
#        current = ttvalue._value
#        if isinstance(current,tree.TTTensorEval):
#            return _collect_slices(current._args,include_none_values)
#        elif isinstance(current,tree.TTDerivedTypeMember):
#            result = []
#            while isinstance(current,tree.TTDerivedTypeMember):
#                if isinstance(current._type,tree.TTTensorEval):
#                    result += _collect_slices(current._type._args,include_none_values)
#                if isinstance(current._element,tree.TTTensorEval):
#                    result += _collect_slices(current._element._args,include_none_values)
#                current = current._element
#            return result
#        else:
#            return []
#
#    def is_array_assignment(self,ttlvalue,scope):
#        """
#        :return: If the LHS expression is an identifier and the associated variable
#                 is an array or if the LHS expression is an array colon expression 
#                 such as `A(:,1:n)`.
#        """
#        lvalue_fstr = ttlvalue.fstr()
#        livar = indexer.scope.search_scope_for_var(scope,lvalue_fstr)
#        loop_indices = [] 
#        if livar["rank"] > 0:
#            lvalue_slices_or_none = self._collect_slices_in_ttvalue(ttlvalue,include_none_values=True)
#            lvalue_slices = [r for r in lvalue_slices_or_none if r != None]
#            if len(lvalue_slices_or_none):
#                num_implicit_loops = len(lvalue_slices)
#            else:
#                num_implicit_loops = livar["rank"]
#        return num_implicit_loops > 0
#
#def analyze_assignment(ttassignment):
#    if isinstance(ttassignment,tree.TTAssignment):
#        return
#    pass