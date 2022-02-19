# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from ... import opts
from .. import nodes
from . import accbackends
from gpufort import util


class STAccDirective(nodes.STDirective):
    """Class for handling ACC directives."""

    def __init__(self, first_linemap, first_linemap_first_statement,
                 directive_no):
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   directive_no,
                                   sentinel="!$acc")
        self._default_present_vars = []

    def find_substring(self, token):
        return token in self.first_statement()

    def find_any_substring(self, tokens):
        result = False
        for token in tokens:
            result = result or self.find_substring(token)
        return result

    def find_all_substrings(self, tokens):
        result = True
        for token in tokens:
            result = result and self.find_substring(token)
        return result

    def is_end_directive(self):
        return self.find_substring("acc end")

    def is_data_directive(self):
        return self.find_substring("acc data")

    def is_enter_data_directive(self):
        return self.find_substring("acc enter")

    def is_exit_data_directive(self):
        return self.find_substring("acc exit")

    def is_init_directive(self):
        return self.find_substring("acc init")

    def is_shutdown_directive(self):
        return self.find_substring("acc shutdown")

    def is_update_directive(self):
        return self.find_substring("acc update")

    def is_wait_directive(self):
        return self.find_substring("acc wait")

    def is_loop_directive(self):
        return self.find_substring("acc loop")

    def is_kernels_directive(self):
        return not self.find_substring("acc kernels loop") and\
            self.find_substring("acc kernels")

    def is_parallel_directive(self):
        return not self.find_substring("acc parallel loop") and\
            self.find_substring("acc parallel")

    def is_loop_directive(self):
        return self.find_substring("acc loop")

    def is_parallel_loop_directive(self):
        return self.find_substring("acc parallel loop")

    def is_kernels_loop_directive(self):
        return self.find_substring("acc kernels loop")

    def is_declare_directive(self):
        return self.find_substring("acc declare")

    def is_purely_declarative(self):
        return self.is_declare_directive() or\
               self.find_substring("acc routine")

    def __str__(self):
        return """
{{ single_line_statement={single_line_statement},
         is_init_directive={is_init_directive},
         is_shutdown_directive={is_shutdown_directive},
         is_end_directive={is_end_directive},
         is_enter_data_directive={is_enter_data_directive},
         is_exit_data_directive={is_exit_data_directive},
         is_wait_directive={is_wait_directive},
         is_loop_directive={is_loop_directive},
         is_parallel_directive={is_parallel_directive},
         is_kernels_directive={is_kernels_directive},
         is_parallel_loop_directive={is_parallel_loop_directive} }}
""".format(single_line_statement=self.first_statement(),
           is_init_directive=self.is_init_directive(),
           is_shutdown_directive=self.is_shutdown_directive(),
           is_end_directive=self.is_end_directive(),
           is_enter_data_directive=self.is_enter_data_directive(),
           is_exit_data_directive=self.is_exit_data_directive(),
           is_wait_directive=self.is_wait_directive(),
           is_loop_directive=self.is_loop_directive(),
           is_parallel_directive=self.is_parallel_directive(),
           is_kernels_directive=self.is_kernels_directive(),
           is_parallel_loop_directive=self.is_parallel_loop_directive()).strip(
           ).replace("\n", "")

    __repr__ = __str__

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if self.is_purely_declarative():
            return nodes.STNode.transform(self, joined_lines,
                                          joined_statements,
                                          statements_fully_cover_lines, index)
        else:
            checked_dialect = nodes.check_destination_dialect(
                opts.destination_dialect)
            return accbackends.ACC_BACKENDS[checked_dialect](self).transform(\
                    joined_lines,joined_statements,statements_fully_cover_lines,index)


class STAccLoopNest(STAccDirective, nodes.STLoopNest):

    def __init__(self, first_linemap, first_linemap_first_statement,
                 directive_no):
        STAccDirective.__init__(self, first_linemap,
                                first_linemap_first_statement, directive_no)
        nodes.STLoopNest.__init__(self, first_linemap,
                                  first_linemap_first_statement)

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[],
                  destination_dialect=""):
        """
        :param destination_dialect: allows to override default if this kernel
                                   should be translated via another backend.
        """
        checked_dialect = nodes.check_destination_dialect(\
            opts.destination_dialect if not len(destination_dialect) else destination_dialect)
        return accbackends.ACC_LOOP_KERNEL_BACKENDS[checked_dialect](self).transform(\
                joined_lines,joined_statements,statements_fully_cover_lines,index)
