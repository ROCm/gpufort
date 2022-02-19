# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import translator
from gpufort import util
from ... import opts
from . import cufbackends


class CufLoopNest2Omp(cufbackends.CufBackendBase):

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        """Analyze based on statements but modify original lines if these are
        fully covered by the statements.
        """
        try:
            parent_tag = self._stnode.parent.tag()
            scope = scope.create_scope(index, parent_tag)
            parse_result = translator.parse_loop_kernel(
                joined_statements.splitlines(), scope)
            f_snippet = joined_lines if statements_fully_cover_lines else joined_statements
            return parse_result.omp_f_str(f_snippet), True
        except Exception as e:
            util.logging.log_exception(LOG_PREFIX, "CufLoopNest2Omp.transform",
                                       "failed to parse loop kernel")
            sys.exit(2) # TODO error code


cufbackends.register_cuf_backend("omp", CufLoopNest2Omp)
