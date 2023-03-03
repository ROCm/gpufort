from gpufort import util

from .. import tree

from . import base

class LoopTransformer(base.Transformer):
    
    def __init__(self):
        base.Transformer.__init__(self)
        self.fun_loop_length = "gpufort::loop_len"
        self.fun_outermost_index = "gpufort::outermost_index"

    def _loop_length(self, loop):
        result = tree.TTCFunctionCall(
            self.fun_loop_length, [loop.first, loop.last]
        )
        if loop.step != None:
            result.args.append(loop.step)
        return result

    def _render_index_recovery(self, first, step, normalized_index):
        """:return: Expression that computes an original loop index
        from the normalized index, which runs from 0 to the length
        of the original loop, and the `first` index and `step` size
        of the original loop. 
        """
        if first != "0":
            result = first+" + "
        else:
            result = ""
        if step != None:
            result += "({})*{}".format(
                step,
                normalized_index
            )
        else:
            result += normalized_index
        return result

    def _tile(self, loop, tile_size: str, tile_loop_index_var=None):
        """
        :param tile_loop_index_var: Index to use for the loop over the tiles,
                                    chosen automatically if None is passed.
        """
        assert loop(loop, (tree.TTCForLoop, tree.TTDo))
        # tile loop
        subst = tree.TTSubstContainer(loop)
        orig_len_var = self.unique_c_identifier("len")
        num_tiles_var = self.unique_c_identifier("num_tiles")
        element_loop_index_var = self.unique_c_identifier("elem")
        normalized_index_var = self.unique_c_identifier("idx")

        num_tiles = "gpufort::div_round_up({loop_len},{tile_size})".format(
            loop_len=orig_len_var,
            tile_size=tile_size
        )
        subst.body += [
            tree.TTCVarDecl("int", orig_len_var,
                            self._loop_length(loop), "const"),
            tree.TTCVarDecl("int", num_tiles_var, num_tiles, "const"),
        ]
        if tile_loop_index_var == None:
            tile_loop_index_var = self.unique_c_identifier("tile")
            subst.body.append(
                tree.TTCVarDecl("int", tile_loop_index_var),
            )
        tile_loop = tree.TTCForLoop(
            tile_loop_index_var,
            loop.excl_ubound,
            incl_lbound=loop.incl_lbound,
            step=None,
        )
        # element_loop
        normalized_index = "{} + ({})*{}".format(
            element_loop_index_var,
            tile_size,
            tile_loop_index_var
        )
        element_loop = tree.TTCForLoop(
            element_loop_index_var,
            tile_size,
            first=tree.TTNumber(["0"]),
            step=loop.step,
        )
        tile_loop.body.append(tree.TTCVarDecl("int", element_loop_index_var))
        # element loop body prolog
        element_loop.body += [
            tree.TTCVarDecl("int", normalized_index_var,
                            normalized_index, "const"),
            tree.TTCElseIf(f"{normalized_index} < {orig_len_var}"),
        ]
        element_loop.body[-1].body.append(
            tree.TTAssignment(
                loop.index,
                self._render_index_recovery(
                    loop.first, loop.step, normalized_index_var
                )
            )
        )
        element_loop.body[-1] += loop.body
        return (subst, tile_loop, element_loop)

    def _collect_n_loops(self,
                         container: tree.TTContainer,
                         n: int,
                         perfectly_nested: bool):
        """Collect `n` loops within the container (container
        might be loop itself).

        Args:
            container: A container statement.
            n (int): Number of loops to collect.
            perfectly_nested (bool): The n loops must be perfectly nested.

        Raises:
            util.error.TransformationError: If the loopnest is interrupted
                                            by a container statement,
                                            or by any other statement and 
                                            perfectly_nested is set to `True`.
                                            If not enough loops could be found.

        Returns:
            (list,list): A tuple of of the loops and other statements that
            separate the loops from each other.
        """
        loops = []
        other_statements = []
        for stmt in container.walk_statements_preorder():
            if isinstance(stmt, tree.TTContainer) and not stmt.has_single_child:
                raise util.error.TransformationError("no loopnest")
            if isinstance(stmt, (tree.TTDo, tree.TTCForLoop)):
                loops.append(stmt)
            elif not perfectly_nested:
                other_statements.append(stmt)
            else:
                raise util.error.TransformationError("not perfectly nested")
            if len(loops) == n:
                break
        if n <= len(loops):
            raise util.error.TransformationError("did only find {} loops instead of {}".format(len(loops),n))
        return (loops, other_statements)

    def tile(self, first, tile_sizes: list(str)) -> tuple:
        """Tile a loopnest according to the given tile sizes.

        Args:
            first: The first loop. TTCForLoop or TTDo.
            tile_sizes (list): Expressions for the tile sizes.

        Returns:
            tuple(tree.TTSubstContainer,tree.TTCForLoop): First component contains the nested loops as TTCForLoop and
               plus all helper variables that have been introduced, Second component returns the first loop.
        """
        assert len(tile_sizes) > 0
        # collect loops
        (loops, _) = self._collect_n_loops(first, len(tile_sizes), False)
        # start from the innermost
        remaining_tile_sizes = list(tile_sizes)
        tile_loops, element_loops = [], []
        (subst, tile_loop, element_loop) = self._tile(loops[-1], remaining_tile_sizes.pop(-1))
        tile_loops.append(tile_loop)
        element_loops.append(element_loop)
        for loop in reversed(loops[:-1]):
            loop.body[0] = subst
            (subst, tile_loop, element_loop) = self._tile(loop, remaining_tile_sizes.pop(-1))
        return (subst, tile_loops, element_loops)

    def collapse(self, first, n: int, force: bool) -> tuple(tree.TTSubstContainer,tree.TTCForLoop):
        """Collapses a nest of n loops

        Args:
            self (_type_): _description_
            tree (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(first, (tree.TTDo, tree.TTCForLoop))
        if n <= 1:
            return first
        subst = tree.TTSubstContainer(first)
        (loops, intermediate_statements) = self._collect_n_loops(first, n, not force)
        # loop length
        loop_len_vars = []
        for loop in loops:
            loop_len_vars.append(self.unique_c_identifier("len"))
            subst.append(
                tree.TTCVarDecl(
                    "int", loop_len_vars[-1], self._loop_length(loop), "const"
                )
            )
        total_len_var = self.unique_c_identifier("total_len")
        subst.append(
            tree.TTCVarDecl(
                "int", total_len_var,
                tree.TTBinaryOpChain.multiply(loop_len_vars), "const"
            )
        )
        # collapsed index and loop
        collapsed_index_var = self.unique_c_identifier("idx")
        subst.append(tree.TTCVarDecl("int", collapsed_index_var))
        collapsed_loop = tree.TTCForLoop(
            collapsed_index_var,
            total_len_var
        )
        subst.append(collapsed_loop)
        # collapsed loop body
        remainder_var = self.unique_c_identifier("rem")
        denominator_var = self.unique_c_identifier("denom")
        collapsed_loop.body += [
            tree.TTCVarDecl("int", remainder_var, collapsed_index_var),
            tree.TTCVarDecl("int", denominator_var, total_len_var)
        ]
        # recover original index
        for i, loop in enumerate(loops):
            assert isinstance(loop, (tree.TTCForLoop, tree.TTDo))
            func_call = tree.TTCFunctionCall(
                remainder_var,
                denominator_var,
                loop.incl_lbound,
                loop_len_vars[i]
            )
            if loop.step != None:
                func_call.args.append(loop.step)
            collapsed_loop.body.append(
                tree.TTCAssignment(loop.index, func_call
                                           ))
        # add intermediate statements
        collapsed_loop.body += intermediate_statements
        return (subst, collapsed_loop)