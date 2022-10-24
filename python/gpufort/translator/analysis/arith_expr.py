# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import indexer 
from .. import tree

from . import fortran

def assign_type_information(ttarith_expr,scope):
    """Traversal action that collects lvalues and rvalues, 
    excludes numbers and characters."""
    if isinstance(expr,tree.TTValue):
        if isinstance(expr._value,(tree.TTNumber,
                                 tree.TTCharacter)):
          pass
      elif isinstance(expr,(tree.TTRvalue,tree.TTLvalue):
          pass
    
        
def search_scope_for_value(ttvalue,scope):
    pass
