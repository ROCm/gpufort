# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import enum

from .. import tree

#def _visit_values(expr,parents,lvalues,rvalues):
#  """Traversal action that collects lvalues and rvalues, 
#  excludes numbers and characters."""
#  if isinstance(expr,tree.TTValue):
#      if isinstance(expr._value,(tree.TTNumber,
#                                 tree.TTCharacter)):
#          pass
#      elif isinstance(expr,tree.TTRvalue):
#          rvalues.append(expr)
#      elif isinstance(expr,tree.TTLvalue):
#          lvalues.append(expr)

def find_lvalues_and_rvalues(ttnode,lvalues,rvalues):
    """Collect lvalues and rvalues, exclude numbers and characters."""
    for child in ttnode.walk_preorder():
        if isinstance(child,tree.TTRvalue):
            rvalues.append(child)
        elif isinstance(child,tree.TTLvalue):
            lvalues.append(child)

    #tree.traversals.traverse(
    #    ttnode,
    #    _visit_values,
    #    tree.traversals.no_action,
    #    tree.traversals.no_crit,
    #    lvalues,
    #    rvalues)

class ArithExprInfo:
    """:todo: For the time being, we ignore
    all operations that transform the rank of a subexpression."""
    def __init__(self,ttarithexpr):
        self._ttarithexpr = ttarithexpr
        self._rank = None
        self._top_level_op = None
        self._top_level_rvals = []
            
    def _traverse_main_rvals_and_ops(self,ttnode,action,depth=0):
        if not action(self,ttnode,depth):
            return
        if isinstance(ttnode,(tree.TTUnaryOp,tree.TTBinaryOp)):
            for child in ttnode._expr:
                self._traverse_main_rvals_and_ops(child,action,depth+1)                  

    #def _compute_rank_action(self,ttnode,depth):
    #    if isinstance(tree.TTRvalue):
    #        if isinstance(tree._value,TTIdentifier):
    #    return True
             
    def get_top_level_binary_op(self,ttnode,depth):
        if isinstance(self._ttarithexpr._expr,tree.TTBinaryOp):
            return self._ttarithexpr._expr
        else:
            return None
  
    #@property
    #def rank:
    #    if self._rank == None:
    #        self._traverse_main_rvals_and_ops(
    #            self._ttarithexpr,
    #            self._compute_rank_action)

class AssignmentInfo:

    def __init__(self,ttassignment,scope):
        self._assignment = ttassignment
        self._main_rvals_and_ops = []
        self._scope = scope
        #
        self._lvalue_ivar = self._lookup_lvalue()
        self._implicit_loops_lhs = -1

    def _lookup_lvalue(self):
        lvalue_fstr = self._assignment._lhs.fstr()
        return indexer.scope.search_scope_for_var(scope,lvalue_fstr)

    def _determine_implicit_loops_lhs(self,ttvalue,ivar):
        """:return: If the LHS expression is an identifier and the associated variable
                 is an array or if the LHS expression is an array colon expression 
                 such as `A(:,1:n)`.
        """
        loop_indices = [] 
        if ivar["rank"] > 0:
            lvalue_ranges_or_none = self._collect_ranges_in_ttvalue(ttlvalue,include_none_values=True)
            lvalue_ranges = [r for r in lvalue_ranges_or_none if r != None]
            if len(lvalue_ranges_or_none):
                num_implicit_loops_lhs = len(lvalue_ranges)
            else:
                num_implicit_loops_lhs = livar["rank"]
        return num_implicit_loops_lhs

    def _collect_ranges(self,function_call_args,include_none_values=False):
        ttranges = []
        for i,ttnode in enumerate(function_call_args):
            if isinstance(ttnode, tree.TTSlice):
                ttranges.append(ttnode)
            elif include_none_values:
                ttranges.append(None)
        return ttranges
    
    def _collect_ranges_in_ttvalue(self,ttvalue,include_none_values=False):
        """
        :return A list of range objects. If the list is empty, no function call
                or tensor access has been found. If the list contains a None element
                this implies that a function call or tensor access has been found 
                but a scalar index argument was used.
        """
        current = ttvalue._value
        if isinstance(current,tree.TTFunctionCall):
            return self._collect_ranges(current._args,include_none_values)
        elif isinstance(current,tree.TTDerivedTypePart):
            result = []
            while isinstance(current,tree.TTDerivedTypePart):
                if isinstance(current._type,tree.TTFunctionCall):
                    result += self._collect_ranges(current._type._args,include_none_values)
                if isinstance(current._element,tree.TTFunctionCall):
                    result += self._collect_ranges(current._element._args,include_none_values)
                current = current._element
            return result
        else:
            return []

    def is_full_array_initialization_with_scalar_rhs(self):
        """Is an expression of the form
        a = 1
        a(:,:) = b * 2*c ! b,c: main operators, scalar
        a(:,:) = b(i) * 2*c(j) ! b(i),c(j): main operators, scalar
        """
        pass

    def is_full_array_reduction_to_scalar(self):
        pass    

    def is_partial_array_reduction_to_scalar(self):
        pass
 
    @property
    def is_array_initialization_with_scalar(self):
        if self._implicit_loops_lhs < 0: # unitialized
            self._implicit_loops_lhs = self._determine_implicit_loops_lhs(
              self._assignment._lhs,
              self._lvalue_ivar
            )
        return self._implicit_loops_lhs > 0

    @property
    def is_array_assignment(self):
        if self._implicit_loops_lhs < 0: # unitialized
            self._implicit_loops_lhs = self._determine_implicit_loops_lhs(
              self._assignment._lhs,
              self._lvalue_ivar
            )
        return self._implicit_loops_lhs > 0

    

    #def _identify_assignment_type(self,scope):
    #    if self._is_array_assignment(self._assignment.lhs,scope):
    #        self._type = AssignmentType.LHS_ARRAY   
    #    elif self._is_array_reduction_intrinsic_call(self._assignment.lhs,scope):
    #        if self._lvalue_ivar
    
    def _is_array_reduction_intrinsic_call(self,ttassignment,scope):
        """
        x = OP(ARRAY,args)
        where OP is one of
        SUM(SOURCE[,DIM][,MASK]) -- sum of array elements (in an optionally specified dimension under an optional mask).
        MAXVAL(SOURCE[,DIM][,MASK]) -- maximum Value in an array (in an optionally specified dimension
                                       under an optional mask);
        MINVAL(SOURCE[,DIM][,MASK]) -- minimum value in an array (in an optionally specified dimension
                                      under an optional mask); 
        ALL(MASK[,DIM]) -- .TRUE. if all values are .TRUE., (in an optionally specified dimension);
        ANY(MASK[,DIM]) -- .TRUE. if any values are .TRUE., (in an optionally specified dimension);
        COUNT(MASK[,DIM]) -- number of .TRUE. elements in an array, (in an optionally specified dimension);
        """
        reduction_ops = {}
        #value_type, index_record = indexer.scope.search_scope_for_value_expr(scope, ident)
        #if value_type == indexer.indexertypes.ValueType.VARIABLE:
        #    value._value._type = tree.TTFunctionCall.Type.ARRAY_ACCESS
        #elif value_type == indexer.indexertypes.ValueType.PROCEDURE:
        #    value._value._type = tree.TTFunctionCall.Type.FUNCTION_CALL
        #elif value_type == indexer.indexertypes.ValueType.INTRINSIC:
        #    value._value._type = tree.TTFunctionCall.Type.INTRINSIC_CALL
        return False    

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
#                if isinstance(ttnode,(tree.TTTTOperator,tree.TTRvalue)):
#                    self._main_rvals_and_ops.append(ttnode)
#                elif isinstance(ttnode,pyparsing.ParseResults):
#                    for child in ttnode._expr:
#                        if isinstance(child,(tree.TTTTOperator,tree.TTRvalue)):
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
#        if isinstance(current,tree.TTFunctionCall):
#            return _collect_slices(current._args,include_none_values)
#        elif isinstance(current,tree.TTDerivedTypePart):
#            result = []
#            while isinstance(current,tree.TTDerivedTypePart):
#                if isinstance(current._type,tree.TTFunctionCall):
#                    result += _collect_slices(current._type._args,include_none_values)
#                if isinstance(current._element,tree.TTFunctionCall):
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
#                num_implicit_loops_lhs = len(lvalue_slices)
#            else:
#                num_implicit_loops_lhs = livar["rank"]
#        return num_implicit_loops_lhs > 0
#
#def analyze_assignment(ttassignment):
#    if isinstance(ttassignment,tree.TTAssignment):
#        return
#    pass