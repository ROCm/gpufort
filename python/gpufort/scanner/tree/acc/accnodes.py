# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from ... import opts

from .. import nodes

class STAccDirective(nodes.STDirective):
    """Class for modeling/handling ACC directives."""

    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, singleton):
        cls._backends.append((dest_dialects, singleton))

    def __init__(self, first_linemap, first_linemap_first_statement,
                 directive_no):
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   directive_no,
                                   sentinel="!$acc")
        self.dest_dialect = opts.destination_dialect
        #
        _, self.directive_kind, self.directive_args, unprocessed_clauses = util.parsing.parse_acc_directive(self.first_statement())
        self.clauses = util.parsing.parse_acc_clauses(unprocessed_clauses)

    def is_directive(self,kind=[]):
        """:return if this is a directive of the given kind,
                   where kind is a list of directive_parts such as 
                   ['acc','enter','region'].
        """
        return self.directive_kind == kind 

    def get_matching_clauses(self,clause_kinds):
        """:return: List of clauses whose kind is part of
        `clause_kinds`.
        :param list clause_kinds: List of clause kinds in lower case.
        """
        return [clause for clause in self.clauses
                if clause[0].lower() in clause_kinds] 

    def is_purely_declarative(self):
        return (self.is_directive(["acc","declare"])
               or self.is_directive(["acc","routine"]))

    def get_async_clause_queue(self):
        """:return: Tuple of the argument of the async clause or None
                    and a bool if the clause is present.
        :raise util.error.SyntaxError: If the async clause appears more than once or if
                                       it has more than one argument.
        """
        async_clauses = self.get_matching_clauses(["async"]) 
        if len(async_clauses) == 1:
            _,args = async_clauses[0]
            if len(args) == 1:
                return args[0], True
            elif len(args) > 1:
                raise util.error.SyntaxError("'async' clause may only have one argument")
            else:
                return None, True
        elif len(async_clauses) > 1:
            raise util.error.SyntaxError("'async' clause may only appear once")
        else:
            return None, False
    
    def get_wait_clause_queues(self):
        """:return: Tuple of the argument of the wait clause or None
                    and a bool if the clause is present. 
        Wait clause may appear on parallel, kernels, or serial construct, or an enter data, exit data, or update directive
        :raise util.error.SyntaxError: If the async clause appears more than once or if
                                       it has more than one argument.
        """
        wait_clauses = self.get_matching_clauses(["wait"]) 
        if len(wait_clauses) == 1:
            _,args = wait_clauses[0]
            return args, True
        elif len(wait_clauses) > 1:
            raise util.error.SyntaxError("'wait' clause may only appear once")
        else:
            return None, False

    def has_finalize_clause(self):
        """:return: If a finalize clause is present
        :raise util.error.SyntaxError: If the finalize clause appears more than once or if
                                       it has arguments.
        """
        finalize_clauses = self.get_matching_clauses(["finalize"]) 
        if len(finalize_clauses) == 1:
            _,args = util.parsing.parse_acc_clauses(finalize_clauses)[0]
            if len(args):
                raise util.error.SyntaxError("'finalize' clause does not take any arguments")
            else:
                return True
        elif len(finalize_clauses) > 1:
            raise util.error.SyntaxError("'finalize' clause may only appear once")
        else:
            return False

    def get_if_clause_condition(self):
        """:return: Empty string if no if was found
        :rtype: str
        """
        if_clauses = self.get_matching_clauses(["if"]) 
        if len(if_clauses) == 1:
            _, args = if_clauses[0]
            if len(args) == 1:
                return args[0], True
            else:
                raise util.error.SyntaxError("'if' clause must have single argument")
        elif len(if_clauses) > 1:
            raise util.error.SyntaxError("'if' clause may only appear once")
        else:
            return None, False

    def transform(self,*args,**kwargs):
        if self.is_purely_declarative():
            return nodes.STNode.transform(self,*args,**kwargs)
        else:
            for dest_dialects, singleton in self.__class__._backends:
                if self.dest_dialect in dest_dialects:
                    singleton.configure(self)
                    return singleton.transform(*args,**kwargs)


class STAccLoopNest(STAccDirective, nodes.STLoopNest):
    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, singleton):
        cls._backends.append((dest_dialects, singleton))

    def __init__(self, first_linemap, first_linemap_first_statement,
                 directive_no):
        STAccDirective.__init__(self, first_linemap,
                                first_linemap_first_statement, directive_no)
        nodes.STLoopNest.__init__(self, first_linemap,
                                  first_linemap_first_statement)
        self.dest_dialect = opts.destination_dialect

    def get_vars_present_per_default(self):
        """:return: If unmapped variables are present by default.
        :raise util.error.SyntaxError: If the present clause appears more than once or if
                                       it more than one argument or .
        """
        default_clause = next((c for c in self.clauses if c[0].lower()=="default"),None)
        if default_clause == None:
            return True
        elif len(default_clause[1]) == 1:
            if len(default_clause[1]) > 1:
                raise util.error.SyntaxError("OpenACC 'default' does only take one argument")
            value = default_clause[1][0].lower()
            if value == "present":
              return True
            elif value == "none":
              return False
            else:
                raise util.error.SyntaxError("OpenACC 'default' clause argument must be either 'present' or 'none'")
        else:
            raise util.error.SyntaxError("only a single OpenACC 'default' clause argument must be specified")

    def transform(self,*args,**kwargs):
        for dest_dialects, singleton in self.__class__._backends:
            if self.dest_dialect in dest_dialects:
                singleton.configure(self)
                return singleton.transform(*args,**kwargs)
