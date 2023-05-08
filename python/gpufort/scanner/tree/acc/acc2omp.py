
from gpufort import indexer
from gpufort import translator
from gpufort import util

from ... import opts

from .. import backends

from . import accbackends
from . import accnodes

dest_dialects = ["omp"]
backends.supported_destination_dialects.update(set(dest_dialects)) 

class Acc2Omp(accbackends.AccBackendBase):

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if not opts.translate_other_directives:
            return (None,False)
        snippet = joined_statements
        try:

            def repl(parse_result):
                return parse_result.omp_f_str(), True
            result,_ = util.pyparsing.replace_first(snippet,\
                     translator.tree.grammar.acc_simple_directive,\
                     repl)
            return result, True
        except Exception as e:
            util.logging.log_exception(
                opts.log_prefix, "Acc2Omp.transform",
                "failed parse directive " + str(snippet))


class AccComputeConstruct2Omp(accbackends.AccBackendBase):

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if not opts.translate_compute_constructs:
            return (None,False)
        parent_tag = self.stnode.parent.tag()
        scope      = indexer.scope.create_scope(index, parent_tag)
        ttcomputeconstruct = self.stnode.parse_result 
        
        arrays       = translator.analysis.arrays_in_subtree(ttcomputeconstruct, scope)
        inout_arrays = translator.analysis.inout_arrays_in_subtree(ttcomputeconstruct, scope)

        snippet = joined_lines if statements_fully_cover_lines else joined_statements
        return translator.codegen.translate_loopnest_to_omp(snippet, ttcomputeconstruct, inout_arrays, arrays), True

accnodes.STAccDirective.register_backend(dest_dialects,Acc2Omp())
accnodes.STAccComputeConstruct.register_backend(dest_dialects,AccComputeConstruct2Omp())
