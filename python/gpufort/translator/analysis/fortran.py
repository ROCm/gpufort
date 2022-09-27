# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
def _visit_values(expr,parents,lvalues,rvalues)
  """Traversal action that collects lvalues and rvalues, 
  excludes numbers and characters."""
  if isinstance(expr,TTValue):
      if isinstance(expr._value,(tree.TTNumber,
                                 tree.TTCharacter)):
          pass
      elif isinstance(expr,TTRvalue):
          rvalues.append(expr)
      elif isinstance(expr,TTLvalue):
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
    
