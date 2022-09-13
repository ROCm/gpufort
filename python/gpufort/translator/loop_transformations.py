import copy
import textwrap

from . import tree
   
single_level_indent = " "*2

def reset():
  _counters.clear()

hip_kernel_prolog =\
""""\
gpufort::acc_triple _res(gridDim.x,div_round_up(blockDim.x/warpSize),{max_vector_length});
gpufort::acc_triple _coords(gang_id,worker_id,vector_lane_id);
"""

_counters = {}
 
def _unique_label(label: str):
    """Returns a unique label for a loop variable that describes
    a loop entity. The result is prefixed with "_" to
    prevent collisions with Fortran variables.
    """
    if label not in _counters:
        _counters[label] = 0
    counter = _counters[label]
    _counters[label] += 1
    return "_"+label+str(result)

def _render_affine_transform(a,b,c):
    """:return: String expression '{a} + ({b})*({c})'."""
    return "{a} + ({b})*({c})".format(
        a=a,b=b,c=c)

def _render_for_loop_open(index,incl_lbound,excl_ubound,step=None):
  if step == None:
      template = """\
for (int {0} = {1}; 
         {0} < {2}; {0}++) {
"""
  else:
      template = """\
for (int {0} = {1};
         {0} < {2}; {0} += {3}) {
"""
  return template.format(index,incl_lbound,excl_ubound,step)


_hip_loop_prolog =\
"""
gpufort::acc_triple {local_res}({num_gangs},{num_workers},{vector_length});
if ( _coords < {local_res} ) {
"""

_hip_loop_epilog = """"\
} // {local_res}
"""

class Loop:
    
    def __init__(self,
          index = None,
          first = None,
          last = None,
          length = None,
          excl_ubound = None,
          step = None,
          gang_partitioned = False,
          worker_partitioned = False,
          vector_partitioned = False,
          num_gangs = "1",
          num_workers = "1",
          vector_length = "1",
          prolog = None,
          body_prolog = None):
        self.index = index
        self.first = first
        self._last = last
        self._length = length
        self.step = step
        self.gang_partitioned = gang_partitioned
        self.worker_partitioned = worker_partitioned
        self.vector_partitioned = vector_partitioned
        self.num_workers = num_workers
        self.num_gangs = num_gangs
        self.vector_length = vector_length
        self.prolog = prolog
        self.body_prolog = body_prolog
        self.body_indent_levels = 1

    def last(self):
        assert (self._length != None 
               or self._last != None
               or self._excl_ubound != None)
        if self._last != None:
            return self._last
        elif self._excl_ubound != None:
            return "({} - 1)".format(self._excl_ubound)
        else self._length != None:
            return "({} + {} - 1)".format(self.first,self._length)
        
    def excl_ubound(self):
        assert (self._length != None 
               or self._last != None
               or self._excl_ubound != None)
        if self._excl_ubound != None:
            return self._excl_ubound
        elif self._last != None:
            return "({} + 1)".format(self._last)
        elif self._length != None:
            return "({} + {})".format(self.first,self._length)

    def length(self):
        assert (self._length != None 
               or self._last != None
               or self._excl_ubound != None)
        if self.length != None:
            return self.length
        else:
            gpufort_fun = "gpufort::loop_len"
            if self._step == None:
                return "{}({},{})".format(
                  gpufort_fun,
                  self._first,self.last())
            else:
                return "{}({},{},{})".format(
                  gpufort_fun,
                  self._first,self.last(),self._step)
    
    def tile(self,tile_size: str):
        gpufort_fun = "gpufort::divide_round_up"
        # tile loop
        orig_len_var = _unique_label("len")
        tile_loop_prolog = "const int {orig_len} = {rhs};".format(
            orig_len = orig_len_var,
            rhs=self.length()
        )
        num_tiles = "{}({},{})".format(gpufort_fun,
                                       orig_len_var,tile_loop_size)
        tile_loop_index = _unique_label("tile")
        tile_loop = Loop(
            index = tile_loop_index;
            first = "0"
            length = num_tiles
            excl_ubound = num_tiles
            step = None
            gang_partitioned = self.gang_partitioned
            worker_partitioned = not self.vector_partitioned and self.worker_partitioned,
            vector_partitioned = self.vector_partitioned,
            num_gangs = "1",
            num_workers = self.num_workers if not self.vector_partitioned else,
            vector_length = "1",
            prolog=tile_loop_prolog)
        # element_loop
        element_loop_index = _unique_label("elem")
        normalized_index_var = _unique_label("idx")
        element_loop_body_prolog += "const int {normalized_index} = {rhs};\n".format(
            normalized_index=normalized_index_var,
            rhs=_render_affine_transform(
              element_loop_index,tile_loop_size,tile_loop_index)
        )
        if self._step != None:
            loop_index_recovery = _render_affine_transform(
              self.first,self.step,normalized_index)
        else:
            loop_index_recovery = self.first 
        element_loop_body_prolog += "if ( {normalized_index} < {orig_len} ) {\n".format(
          normalized_index=normalized_index_var,
          orig_len=orig_len_var
        )
        element_loop_body_prolog += single_level_indent
        element_loop_body_prolog +"const int {original_index} = {rhs};\n".format(
          original_index=self.index,
          rhs=loop_index_recovery
        )
        element_loop = Loop(
            index = element_loop_index
            first = "0"
            length = tile_loop_size
            excl_ubound = tile_loop_size,
            step = self._step
            gang_partitioned = self.gang_partitioned
            worker_partitioned = not self.vector_partitioned and self.worker_partitioned,
            vector_partitioned = self.vector_partitioned,
            num_gangs = "1",
            num_workers = self.num_workers if not self.vector_partitioned else,
            vector_length = self.vector_length,
            body_prolog = element_loop_body_prolog)
        element_loop.body_indent_levels = 2
        return (tile_loop,element_loop)

    def map_to_hip_cpp(self):
        """
        :return: HIP C++ device code.
        """
        resources = []
        
        indent = ""
        loop_open = self.prolog
        partitioned = (
          self.gang_partitioned
          or self.worker_partitioned
          or self.vector_partitioned
        )
  
        if partitioned: 
            local_res_var = _unique_label("local_res")
            loop_open += _hip_loop_prolog.format(
              local_res=local_res_var
              num_gangs=self.num_gangs,
              num_workers=self.num_workers,
              vector_length=self.vector_length
            )
            tile_size_var = _unique_label("tile_size")
            num_tiles = "{}.product()".format(local_res_var)
            gpufort_fun = "gpufort::div_round_up"
            loop_open += "const int {tile_size} = {fun}({len},{tiles});\n".format(
                tile_size_var,fun=gpufort_fun,
                n=self.length(),num_tiles)
            # deep copy this loop and modify 
            local_id_var = _unique_label("local_id")
            loop_open += \
              "const int {local_id} = _coords.linearize({local_res});\n".format(
                local_id=local_id_var,
                local_res=local_res_var
              )
            loop_open += self.prolog
            loop = copy.deepcopy(self)
            #
            if self.vector_partitioned: # vector, worker-vector, gang-worker-vector
                incl_lbound = "{} + {}".format(self.first,local_id_var)
                step = tile_size_var
                loop_open +=_render_for_loop_open(\
                  self.index,first,self.excl_ubound,step) 
                loop_open  += self.body_prolog.format(idx=local_id_var)
            else: # gang, gang-worker
                incl_lbound = "{} + {}*{}".format(self.first,local_id_var,tile_size_var)
                excl_ubound = "{} + {}*({}+1)"
                loop_open  += self.body_prolog.format(idx=local_id_var)
              
                # take element_loop
                pass                   
            loop_close += _hip_loop_epilog.format(local_res_var)
        else:
            loop_open += _render_for_loop_open(\
              self.index,self.first,self.excl_ubound(),self.step),
            loop_close = "}\n"
        # add the body prolog, outcome of previous loop transformations
        indent += single_level_indent
        loop_open += textwrap.indent(self.body_prolog.format(idx=self.index,indent)
        return (loop_open,loop_close)

class Loopnest:
    """ Transforms tightly-nested loopnests where only the first loop
    stores information about loop transformations and mapping the loop to a HIP
    device. Possible transformations are collapsing and tiling of the loopnest.
    In the latter case, the loopnest must contain as many loops as tiles.
    Mapping to a HIP device is performed based on the
    offload information stored in the first loop of the loopnest.
    """
    def __init__(self):
        self._loops = []
        self._original_loops = []
        self._is_tiled = False
    def append_loop(self,loop):
        self._loops.append(loop) 
        self._original_loops.append(loop)
    def collapse(self):
        assert !self._is_tiled, "won't collapse tiled loopnest"
        assert len(self._loops)
        loop_lengths_vars = []
        first_loop = self._loops[0]
        # Preamble before loop
        prolog = ""
        for i,loop in enumerate(self._loops):
            loop_lengths_vars.append(_unique_label("len"))
            prolog += "const int {} = {};\n".format(
              loop_lengths_vars[-1],loop.length())
        total_len_var = _unique_label("total_len")
        prolog += "const int {} = {};\n".format(
              total_len_var, "*".join(loop_lengths_vars)
            )
        # Preamble within loop body
        body_prolog = ""
        remainder_var = _unique_label("rem");
        denominator_var= _unique_label("denom")
        # template, idx remains as placeholder because of double brackets
        body_prolog += "int {rem} = {{idx}};\n".format(
          rem=remainder_var,collapsed_index_var
        )
        body_prolog += "int {denom} = {total_len};\n".format(
          denom=denominator_var,total_len=total_len_var
        )
        # index recovery
        for i,loop in enumerate(self._original_loops):
            gpufort_fun = "gpufort::outermost_index_w_len"
            if loop.step != None:
                body_prolog += ("const int {} = "
                  + gpufort_fun
                  + "({}/*inout*/,{}/*inout*/,{},{},{});\n").format(
                    remainder_var,denominator_var,
                    loop.first,loop_lengths_vars[i],loop.step
                  )
            else:
                body_prolog += ("const int {} = "
                  + gpufort_fun
                  + "({}/*inout*/,{}/*inout*/,{},{});\n").format(
                    remainder_var,denominator_var,
                    loop.first,loop_lengths_vars[i]
                  )
        collapsed_index_var = _unique_label("idx")
        collapsed_loop = Loop(
          index = collapsed_index_var,
          first = "0",
          length = total_len_var,
          excl_ubound = total_len_var,
          step = None,
          gang_partitioned = first_loop.gang_partitioned,
          worker_partitioned = first_loop.worker_partitioned,
          vector_partitioned = first_loop.vector_partitioned,
          num_gangs = first_loop.num_gangs,
          num_workers = first_loop.num_workers,
          vector_length = first_loop.vector_length,
          prolog = prolog, 
          body_prolog = body_prolog)
        self._loops.clear()
        self._loops.append(collapsed_loop)

    def tile(self,tile_sizes):
        if isinstance(tile_sizes,str):
            tile_sizes = [tile_sizes]
        assert len(tile_sizes) == len(self._loops)
        tile_loops = []
        element_loops = []
        for i,loop in enumerate(self._loops()):
            tile_loop, element_loop = loop.tile(tile_sizes[i])
            tile_loops.append(tile_loop)
            element_loops.append(element_loop)
        self._loops.clear()
        self._loops += tile_loops
        self._loops += element_loops
        self._is_tiled = True

    def map_to_hip_cpp(self):
        # TODO analyze and return required resources (gangs,workers,vector_lanes)
        # but perform it outside as we deal with C++ expressions here.
        # * Alternatively, move the derivation of launch parameters into C++ code?
        #   * Problem size depends on other parameters that need to be passed
        #     to the C++ layer anyway ...
        # Think outside of the box:
        # * Gang synchronization might require us to launch multiple kernels
        #   and put wait events inbetween
        # * What about reductions?
        pass
         

# TODO implement
# TODO workaround, for now expand all simple workshares
# Workshares are interesting because the loop
# bounds might actually coincide with the array
# dimensions of a variable
#class Workshare:
#    pass
## Workshare that reduces some array to a scalar, e.g. MIN/MAX/...
#class ReducingWorkshare:
#    pass 
