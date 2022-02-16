# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import translator
from gpufort import util
from .. import opts
from . import acc

class Acc2Omp(acc.AccBackendBase):
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[]):
        
        snippet = joined_statements
        try:
           def repl(parse_result):
                return parse_result.omp_f_str(), True
           result,_ = util.pyparsing.replace_first(snippet,\
                    translator.tree.grammar.acc_simple_directive,\
                    repl)
           return result, True 
        except Exception as e:
           util.logging.log_exception(LOG_PREFIX,"Acc2Omp.transform","failed parse directive "+str(snippet))

class AccLoopNest2Omp(acc.AccBackendBase):
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[]):
        
        if statements_fully_cover_lines:
            snippet = joined_lines
        else:
            snippet = joined_statements
        try:
           parent_tag   = self._stnode.parent.tag()
           scope        = scope.create_scope(index,parent_tag)
           parse_result = translator.parse_loop_kernel(snippet.splitlines(),scope)
           return parse_result.omp_f_str(snippet), True 
        except Exception as e:
           util.logging.log_exception(LOG_PREFIX,"AccLoopNest2Omp.transform","failed to convert kernel "+str(snippet))
           sys.exit(2)

def AllocateOmp(stallocate,index):
    return ""

def DeallocateOmp(stdeallocate,index):
    return ""

acc.register_acc_backend("omp",
                         Acc2Omp,
                         AccLoopNest2Omp,
                         acc.AccPostprocessBackendBase,
                         AllocateOmp,
                         DeallocateOmp,
                         None)