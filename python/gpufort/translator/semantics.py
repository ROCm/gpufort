# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree

#    def is_real_of_kind(self,kind):
#        """:return: If the number is a real (of a certain kind)."""
#        if self._kind == None:
#            return kind == None
#        else:
#            if kind in opts.fortran_type_2_bytes_map["real"]:
#                has_exponent_e = "e" in self._value
#                has_exponent_d = "d" in self._value
#                has_exponent = has_exponent_e or has_exponent_d
#                default_real_bytes = opts.fortran_type_2_bytes_map["real"][
#                    ""].strip()
#                kind_bytes = opts.fortran_type_2_bytes_map["real"][kind].strip()
#                if self._kind == None: # no suffix
#                    cond = False
#                    cond = cond or (has_exponent_d and kind_bytes == "8")
#                    cond = cond or (has_exponent_e and
#                                    kind_bytes == default_real_bytes)
#                    cond = cond or (not has_exponent and
#                                    kind_bytes == default_real_bytes)
#                    return cond
#                else:
#                    if self._kind in opts.fortran_type_2_bytes_map["real"]:
#                        kind_bytes = opts.fortran_type_2_bytes_map["real"][
#                            self._kind].strip()
#                        return kind_bytes == self._kind_bytes
#                    else:
#                        raise util.error.LookupError(\
#                          "no number of bytes found for kind '{}' in 'translator.fortran_type_2_bytes_map[\"real\"]'".format(self._kind))
#                        sys.exit(2) # todo: error code
#            else:
#                raise util.error.LookupError(\
#                  "no number of bytes found for kind '{}' in 'translator.fortran_type_2_bytes_map[\"real\"]'".format(kind))
#                sys.exit(2) # todo: error code
#        else:
#            return False
#
#    def is_integer_of_kind(self,kind):
#        """:return: If the number is an integer (of a certain kind)."""
#        if not self.type ==  and kind != None:
#            if kind in opts.fortran_type_2_bytes_map["integer"]:
#                has_exponent_e = "e" in self._value
#                has_exponent_d = "d" in self._value
#                has_exponent = has_exponent_e or has_exponent_d
#                default_integer_bytes = opts.fortran_type_2_bytes_map[
#                    "integer"][""].strip()
#                kind_bytes = opts.fortran_type_2_bytes_map["integer"][
#                    kind].strip()
#                if self._kind == None: # no suffix
#                    return kind_bytes == default_integer_bytes
#                else: # suffix
#                    if suffix in opts.fortran_type_2_bytes_map["integer"]:
#                        suffix_bytes = opts.fortran_type_2_bytes_map[
#                            "integer"][suffix].strip()
#                        return kind_bytes == suffix_bytes
#                    else:
#                        raise util.error.LookupError("no number of bytes found for suffix '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(suffix))
#            else:
#                raise util.error.LookupError(\
#                  "no number of bytes found for kind '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(kind))
#        else:
#            return is_integer

def _resolve_literal(literal,scope):
    if isinstance(literal,tree.TTLogical,
                    literal,tree.TTNumber):
        

def _check_derived_type_part(ttderivedtypemember,scope):
    """Check that not more than one part reference has nonzero rank, e.g.
    disallows expressions such as "a(:)%b(:)
    :note: Assumes that all types have already been resolved."""
    already_found_part_reference_with_nonzero_rank = False 
    for ttnode in ttderivedtypemember.walk_derived_type_parts_postorder():
        for child in ttnode.child_nodes():
            if child.rank() > 0:
                if already_found_part_reference_with_nonzero_rank:
                    raise error.SemanticError(
                      "Two or more part references with nonzero rank must not be specified"
                    )
                already_found_part_reference_with_nonzero_rank = True

def _check_complex_arith_expr(ttcomplexarithexpr,scope):
    """Check if a complex type is constructed of compatible components:
    - real and integer are allowed
    - all components must have rank 0
    """ 
    for child in self.child_nodes:
        if child.type not in [tree.Literal.REAL,
                              tree.Literal.INTEGER]:
            raise error.SemanticError("components of complex must be real or integer")
        elif child.rank != 0:
            raise error.SemanticError("rank of complex components must be 0")

def _resolve_tensor_eval(ttnode,scope):
    """
    - resolve type of the expression: function call vs array access
    - in case of function call,
      - check that keyword arguments appear only after positional argument
        - next level: check type of arguments, compare to that of function argument list
      - check if elemental or transformational intrinsic
      - handle conversion functions
    - in case of array access:
      - check that arguments are of integer type
    """
    ttnode.symbol_info = indexer.scope.search_scope_for_value_expr(
      scope,ttnode.fstr()
    )
    if ttnode.is_array_access():
        for arg in ttnode.args():
            
    else: # is function call
        for arg in 
            pass
    assert False, "not implemented"

def _resolve_binary_op(ttbinaryop,scope):
    """
    :note: Assumes that all operands have already been resolved."""
    for ttnode in ttderivedtypemember.operands():
        pass

def resolve_arith_expr(ttarithexpr,scope):
    """Resolve the types in an arithmetic expression parse tree.
    Post-order traversal is mandatory as this traverses descedants
    before their parents.
    :raise util.error.LookupError: if a symbol's type could not be determined.
    :note: Can also be applied to subtrees contained in TTArithExpr.
    """
    for ttnode in ttarithexpr.walk_postorder():
        if isinstance(ttnode,tree.TTIdentifier):
            ttnode.symbol_info = indexer.scope.search_scope_for_var(scope,ttnode.fstr())
            # todo
        elif isinstance(ttnode,tree.TTTensorEval):
            ttnode.symbol_info = indexer.scope.search_scope_for_value_expr(scope,ttnode.fstr())
            # todo
        elif isinstance(ttnode,tree.TTValue):
            if isinstance(ttnode._value,tree.TTDerivedTypePart):
                _check_derived_type_part(ttnode._value,scope)
            