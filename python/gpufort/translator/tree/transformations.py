# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

"""Tree nodes that are introduced in the context of tree transformations.
"""

from . import base

from .. import conv

class TTInjectedNode(base.TTNode):
    def fstr(self):
        assert False, "not implemented"

class TTInjectedStatement(base.TTStatement):
    def fstr(self):
        assert False, "not implemented"

class TTInjectedContainer(base.TTContainer):
    def fstr(self):
        assert False, "not implemented"

class TTDummy(TTInjectedNode):
    """A node with user-defined C/C++ and Fortran
    representations, specified as string.
    """

    def __init__(self,cstr,fstr):
        """:param str cstr: The C/C++ representation.
        :param str fstr: The Fortran representation."""
        self._cstr = cstr
        self._fstr = fstr 
    def cstr(self):
        return self._cstr
    def fstr(self):
        return self._fstr

class TTDummyStatement(TTInjectedStatement):
    """A node with user-defined C/C++ and Fortran
    representations, specified as string or tree node.
    """

    def __init__(self,crepr,frepr):
        """:param crepr: The C/C++ representation.
        :param str frepr: The Fortran representation."""
        self._crepr = crepr
        self._frepr = frepr 
    def cstr(self):
        return self._crepr
    def fstr(self):
        return self._frepr

class TTCDummyStatement(TTInjectedStatement):
    """A node with user-defined C/C++ and NO Fortran
    representations.
    """

    def __init__(self,crepr):
        """:param crepr: The C/C++ representation."""
        self._cstr = crepr
    def cstr(self):
        return self._cstr

class TTUnrolledArrayAssignment(TTInjectedStatement):
    """Specific assignment class for unrolled array assignments."""

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
 
    @property
    def type(self):
        return self.lhs.type
    @property
    def rank(self):
        return 0 
    @property
    def bytes_per_element(self):
        return self.lhs.bytes_per_element
    
    def walk_values_ignore_args(self):
        yield from self.lhs.walk_values_ignore_args()
        yield from self.rhs.walk_values_ignore_args()
    
    def child_nodes(self):
        yield self.lhs
        yield self.rhs
    def cstr(self):
        return base.to_cstr(lhs) + "=" + base.to_cstr(rhs) + ";\n"
    def fstr(self):
        assert False, "no Fortran representation"

class TTSubstStatement(TTInjectedStatement):
    """A statement with original and substituted  
    """

    def __init__(self,orig,subst):
        """Constructor.
        :param str orig: The original node.
        :param str subst: The substitute node.
        """
        self.orig = orig
        self.subst = subst 

    def child_nodes(self):
        yield from self.subst.child_nodes()

    def fstr(self):
        self.orig.fstr()
    
    def cstr(self):
        return base.to_cstr(subst)

class TTSubstContainer(TTInjectedContainer):
    """A container statement that stores the substitution tree in its body
    and stores the original tree in a separate node.
    """

    def __init__(self,orig):
        """Constructor.
        :param str orig: The original node.
        :param str subst: The substitute node.
        """
        self.orig = orig
        self._init(indent="")
    
    def child_nodes(self):
        yield from self.body

    def fstr(self):
        self.orig.fstr()

class TTBracketArrayAccess(TTInjectedNode):
    def __init__(self,name,arg):
        self.name = name
        self.arg = arg
    def cstr(self):    
        return name + "[" + base.to_cstr(arg) + "]"

class TTParenthesesArrayAccess(TTInjectedNode):
    def __init__(self,name,args):
        self.name = name
        self.args = args
    def cstr(self):    
        return (
         name 
         + "(" 
         + ",".join([base.to_cstr(arg) for arg in self.args]) 
         + ")"
        )

class TTCElseIf(TTInjectedContainer):

    def __init__(self,
        condition,
        is_elseif = False
      ):
        self._init()
        self.condition = condition
        self.is_elseif = is_elseif

    def child_nodes(self):
        if isinstance(self.condition,TTNode):
            yield self.condition
        yield from self.body
    
    def header_cstr(self):
        prefix = self.is_elseif+" " if self.is_elseif != None else ""
        return "{}if ({})".format(prefix,base_to_cstr(self.condition))

class TTCForLoop(TTInjectedContainer):
    
    def __init__(self,
          index,
          excl_ubound,
          incl_lbound = None,
          step = None,
          declare_index = False
          index_type = "int"
        ):
          """Except `index_type` all arguments can be 
          and instance of TTNode or str."""
          self._init()
          self.index = index
          self.excl_ubound = excl_ubound
          self.incl_ubound = incl_ubound
          self.step = step
          self.declare_index = declare_index
          self.index_type = index_type

    def header_cstr(self):
        """:return: C++ representation of the loop header.
        :note: Does neither close with line break nor does append '{'.
        """
        if base.to_cstr(self.step) != None:
            step_as_str = base.to_cstr(self.step)
        else:
            step_as_str = "1"
        return """\
for ({prefix}{idx} = {ilb};
     {idx} < {eub}; {idx} += {step_as_str})""".format(
          idx = base.to_cstr(self.index),
          ilb = base.to_cstr(self.incl_lbound) if base.to_cstr(self.incl_lbound) != None else "0",
          eub = base.to_cstr(self.excl_ubound),
          step_as_str = base.to_cstr(self.step) if base.to_cstr(self.step) != None else "1",
          prefix = (self.index_type+" ") if base.to_cstr(self.declare_index) else ""
        )
          
class TTCCopyStatement(TTInjectedStatement):
    """Emits a copy statement such as `dest = src;`
    or `dest[dest_idx] = src[src_idx];`
    """

    def __init__(self,
        dest_name,
        src_name,
        dest_index = None,
        src_index = None,
      ):
        self.dest_name = dest_name
        self.src_name = src_name
        self.dest_index = dest_index
        self.src_index = src_index
    
    def cstr(self):
        dest = self.dest_name
        src = self.src_name
        if self.dest_index != None:
            dest += "[" + self.dest_index + "]"
        if self.src_index != None:
            src += "[" + self.src_index + "]"
        return "{} = {};".format(dest,src)

class TTCCopyForLoop(TTCForLoop):
    """Emits a for loop that copies from one type with 
    operator [] to the other.
    """

    def __init__(self,
        dest_name,
        src_name,
        index,
        num_elements,
      ):
        """:param str dest_name: Name of the buffer to write to.
        :param str src_name: Name of the buffer to read from.
        :param num_elements: The number of elements to copy.
        """
        TTCForLoop.__init__(self,index,num_elements)
        self.body.append(
          TTCCopyStatement(dest_name,src_name,index,index)
        )
  
class TTCVarDecl(TTInjectedStatement):
    """Injects a C/C++ variable declaration.
    """
    
    def __init__(self,
        ctype,
        var_expr,
        rhs = None
      ):
        """Constructor.
        """
        self.type = ctype
        self.var_expr = var_expr
        self.rhs = rhs

    def cstr(self):
        if self.rhs_as_str != None:
            template = "{} {} = {};"
        else:
            template = "{} {};"
        return template.format(
          base.to_cstr(self.type),
          base.to_cstr(self.var_expr),
          base.to_cstr(self.rhs)
        ) 

class TTCArrayVarDecl(TTCVarDecl):
    """Injects a C/C++ array variable declaration.
    """
    
    def __init__(self,
        ctype,
        var_expr = None,
        size_expr = None
        rhs = None
      ):
        """Except `index_type` all arguments can be 
        and instance of TTNode or str."""
        TTCVarDecl.__init__(
          ctype,
          base.to_cstr(var_expr) + "["+ base.to_cstr(size_expr) + "]",
          rhs
        )

class TTCVarDeclFromFortranSymbol(TTCVarDecl):

    def __init__(self,
      name,
      symbol_info,
      rhs = None
    ):
      """
      :param rhs: Either string of C/C++ code or a translator tree node, or None.
      :param bounds: Either list of strings of C/C++ code or a list of translator tree nodes,
                     or None.
      # TODO: Need resolved bounds when resolving clauses.
      """
      type_as_cstr = conv.c_type(
        symbol_info.type,
        symbol_info.bytes_per_element
      )
      if symbol_info.rank > 0:
          var_expr_as_cstr = name + "["+ base.to_cstr(size_expr) + "]",
      else:
          var_expr_as_cstr = name
      #
      TTCVarDecl.__init__(self,
        type_as_cstr,
        var_expr_as_cstr,
        base.to_cstr(rhs) if rhs != None else None
      )
    
  def cstr(self):
      TTCVarDecl.cstr(self)

# todo: deprecated
class TTCReductionVarDeclFromFortranSymbol(TTCVarDeclFromFortranSymbol):
    """Injects reduction variable declarations."""
    # before loopnest

    def __init__(self,
        symbol_info,
        bytes_per_element,
        op,
      ):
        """Constructor.
        :param str op: A Fortran operator. 
        """
        TTCVarDecl.__init__(self,
          symbol_info,
          bytes_per_element)
        self.op = op

    def cstr(self):
        c_type = conv.c_type(
          self.symbol_info.type,
          self.bytes_per_element
        )
        c_init_val = conv.reduction_c_init_val(self.op,c_type)
        return "{} {} = {};".format(c_type,self.name,c_init_val)

class TTCReductionOpApplication(TTInjectedStatement):
    """Injects reduction operator application."""
    # within loopnest, at the end of the inner loop
   
    def __init__(self,
        var_expr,
        buffer_access_expr,
        c_type,
        fortran_op,
      ):
        """Constructor.
        :param str op: A Fortran reduction operator. 
        :param str buffer_access_expr: Expression for accessing the buffer, including
                                       reduction variable etc.
        """
        self.var_expr = var_expr
        self.buffer_access_expr = buffer_access_expr
        self.c_type = c_type
        self.fortran_op = fortran_op

    def cstr(self):
        c_op = conv.reduction_c_op(self.fortran_op)
        if c_op in  ["min","max"]:
            return "{0} = {1}({0},{2});".format(
              base.to_cstr(self.buffer_access_expr),
              c_op,
              base.to_cstr(self.var_expr)
            )
        else:
            return "{0} = {0} {1} {2};".format(
              base.to_cstr(self.buffer_access_expr),
              c_op,
              base.to_cstr(self.var_expr)
            )

class TTCSyncThreads(TTInjectedStatement):
    """Injects CUDA/HIP threadblock synchronization."""
 
    def __init__(self):
        """Constructor."""
        pass

    def cstr(self):
        return "__syncthreads();".format(
          buffer_name,self.op,self.name
        )

class TTCReductionBlockwiseAggregation(TTInjectedContainer):
    """Injects a threadblock-wise aggregation."""
    # e.g. after a loopnest that does a blockwise reduction  

    def __init__(self,
        type_as_cstr,
        op_as_fstr,
        var_write_as_cstr,
        buffer_read_expr_as_cstr,
        buffer_read_index_as_cstr,
        buffer_size_as_cstr,
        mask = "(threadIdx.x + threadIdx.y) == 0"
      ):
        """Constructor."""
        self._init(indent="")
        aggregate_loop = TTCForLoop(
          buffer_read_index_as_cstr,
          buffer_size_as_cstr
        )
        aggregate_loop.append(
          TTReductionOpApplication(
            buffer_access_expr_as_cstr, # notice that buffer and var expr are swapped
            var_expr_as_cstr,
            c_type_as_cstr,
            fortran_op_as_cstr,
          )
        )
        self.body = [
          TTSyncThreads(),
          TTCElseIf(mask),
        ]
        self.body[-1].append(aggregate_loop)

def render_gpufort_array_descr_type(rank,ctype):
    return "gpufort::array_descr{}<{}>".format(rank,ctype)

def render_gpufort_array_type(rank,ctype):
    return "gpufort::array{}<{}>".format(rank,ctype)

class TTCGpufortArrayPtrDecl(TTCVarDecl):
    """Emits C++ code such as `gpufort::array_descr1<float> arr;`.
    """ 
    def __init__(self,
        var_name,
        symbol_info,
      ):
        element_type = conv.c_type(
          symbol_info.type,
          symbol_info.bytes_per_element
        )
        TTCVarDecl.__init__(self,
          render_gpufort_array_descr_type(symbol_info.rank,element_type),
          var_name
        )

class TTCGpufortArrayPtrWrap(TTInjectedStatement):
    """Emits C++ code such as `arr.wrap(&_buffer[0],n1,n2,l1,l2);`.
    """ 
    def __init__(self,
        var_name,
        buffer_name
        symbol_info,
      ):
        """
        :param str buffer_name: Name of the contiguous buffer to access.
        """
        self.var_name = var_name
        self.buffer_name = buffer_name
        self.resolved_bounds = symbol_info.resolved_bounds

    def cstr(self):
        lbounds, sizes = [], []
        for ttextent in self.resolved_bounds:
            lbounds.append(ttextent.lbound.cstr())
            size = ttextent.size_expr
            if size.yields_literal:
                sizes.append(str(size.eval()))
            else:
                sizes.append(size.cstr())
        return "{var}.wrap(nullptr,&{buf}[0],{sizes},{lbounds});".format(
          var = self.var_name,
          buf = self.buffer_name,
          sizes = ",".join(sizes),
          lbounds = ",".join(lbounds),
        )

class TTSubstAccComputeConstruct(TTSubstContainer):
    """Substitution node for an OpenACC compute construct."""
    pass

class TTSubstAccLoopDirective(TTSubstContainer):
    """Substitution node for an OpenACC loop construct."""
    pass
