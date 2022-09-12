import copy

from . import tree
   
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

def reset():
  _counters.clear()

def _render_affine_transform(a,b,c):
    """:return: String expression '{a} + ({b})*({c})'."""
    return "{a} + ({b})*({c})".format(
        a=a,b=b,c=c)

class Loop:
    
    def __init__(self,
          index = None,
          first = None,
          last = None,
          length = None,
          step = None,
          gang_partitioned = False,
          worker_partitioned = False,
          vector_partitioned = False,
          num_workers = 1,
          num_gangs = 1,
          vector_length = 1,
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
    
    def length(self):
        assert self.length != None or self.last != None
        if self.length != None:
            return self.length
        else:
            gpufort_func = "gpufort::loop_len"
            if self._step == None:
                return "{}({},{})".format(
                  gpufort_func,
                  self._first,self._last)
            else:
                return "{}({},{},{})".format(
                  gpufort_func,
                  self._first,self._last,self._step)
    
    def tile(self,tile_size: str):
        gpufort_func = "gpufort::divide_round_up"
        # tile loop
        num_tiles = "{}({},{})".format(gpufort_func,
                                       self.length(),tile_size)
        
        tile_index = _unique_label("tile")
        tile_loop = Loop(
            index = tile_index;
            first = "0"
            length = num_tiles
            step = None
            gang_partitioned = self.gang_partitioned
            worker_partitioned = not self.vector_partitioned and self.worker_partitioned,
            vector_partitioned = self.vector_partitioned,
            num_gangs = "1",
            num_workers = self.num_workers if not self.vector_partitioned else,
            vector_length = "1")
        # element_loop
        element_index = _unique_label("elem")
        normalized_index = _unique_label("idx")
        element_prolog = ""
        element_body_prolog += "const int {} = {};\n".format(
            normalized_index,_render_affine_transform(
              element_index,tile_size,tile_index))
        if self._step != None:
            loop_index_recovery = _render_affine_transform(
              self.first,self.step,normalized_index)
        else:
            loop_index_recovery = self.first 
        element_body_prolog += "const int {} = {};\n".format(
            self.index,loop_index_recovery)
        element_loop = Loop(
            index = element_loop_index
            first = "0"
            length = tile_size
            step = self._step
            gang_partitioned = self.gang_partitioned
            worker_partitioned = not self.vector_partitioned and self.worker_partitioned,
            vector_partitioned = self.vector_partitioned,
            num_gangs = "1",
            num_workers = self.num_workers if not self.vector_partitioned else,
            vector_length = self.vector_length,
            body_prolog = element_body_prolog)
        return (tile_loop,element_loop)

    def map_to_hip_cpp(self,index_override=None):
        resources = []
        if self.gang_partitioned:
            resources.append("("+self.num_gangs+")")
        if self.worker_partitioned:
            resources.append("("+self.num_workers+")")
        if self.vector_partitioned:
            resources.append("("+self.vector_length+")")
        if len(resources):
            tile_size = "*".join(resources) 
            tile_loop, element_loop = self.tile(tile_size)
        else:
            # normal for-loop
            pass

class Loopnest:
    def __init__(self):
        self._loops = []
        self._is_collapsed = False        
        self._is_tiled = False
    def append_loop(self,loop):
        self._loops.append(loop) 

    def collapse(self):
        assert !self._tiled, "won't collapse tiled loopnest"
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
        body_prolog += "int {} = {{idx}};\n".format(
          remainder_var,collapsed_index_var
        )
        body_prolog += "int {} = {};\n".format(
          denominator_var,total_len_var
        )
        # index recovery
        for i,loop in enumerate(self._original_loops):
            gpufort_func = "gpufort::outermost_index_w_len"
            if loop.step != None:
                body_prolog += ("const int {} = "
                  + gpufort_func
                  + "({}/*inout*/,{}/*inout*/,{},{},{});\n").format(
                    remainder_var,denominator_var,
                    loop.first,loop_lengths_vars[i],loop.step
                  )
            else:
                body_prolog += ("const int {} = "
                  + gpufort_func
                  + "({}/*inout*/,{}/*inout*/,{},{});\n").format(
                    remainder_var,denominator_var,
                    loop.first,loop_lengths_vars[i]
                  )
        collapsed_index_var = _unique_label("idx")
        collapsed_loop = Loop(
          index = collapsed_index_var,
          first = "0",
          length = total_len_var,
          step = None,
          gang_partitioned = first_loop.gang_partitioned,
          worker_partitioned = first_loop.worker_partitioned,
          vector_partitioned = first_loop.vector_partitioned,
          num_gangs = first_loop.num_gangs,
          num_workers = first_loop.num_workers,
          vector_length = first_loop.vector_length,
          prolog = prolog, 
          body_prolog = self.body_prolog(collapsed_index_var,))
        self._loops.clear()
        self._loops.append(collapsed_loop)
        self._is_collapsed = True

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

    def map_to_hip_cpp(self):
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
