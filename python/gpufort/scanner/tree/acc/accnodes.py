# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from ... import opts

from .. import nodes

class STAccDirective(nodes.STDirective):
    r""" Class for modeling/handling ACC directives.

    :param parent_directive: The parent directive of this directive
                             or compute construct.
    :type parent_directive: nodes.STAccDirective or None
    :note: Multiple code generation backends can be registered via the `register_backend` class method.
    """

    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, singleton):
        cls._backends.append((dest_dialects, singleton))

    def __init__(self, parent_directive, first_linemap, first_linemap_first_statement):
        """
        :param parent_directive: The parent `acc data` or `acc kernels` directive of this directive
                                 or compute construct. Might be None if no preceding `acc data` or `acc kernels` exists.
        :type parent_directive: nodes.STAccDirective or None
        """
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   sentinel="!$acc")
        self.dest_dialect = opts.destination_dialect
        #
        self.parent_directive = parent_directive
        _, self.directive_kind, self.directive_args, unprocessed_clauses = util.parsing.parse_acc_directive(
                self.first_statement().lower())
        self.clauses = util.parsing.parse_acc_clauses(unprocessed_clauses)
        util.parsing.check_acc_clauses(self.directive_kind,self.clauses)
        # todo: add members documentation

    def is_directive(self,kind=[]):
        """:return if this is a directive of the given kind,
                   where kind is a list of directive_parts such as 
                   ['acc','enter','region'].
        """
        return self.directive_kind == kind 
    
    def get_matching_clauses(self,clause_kinds):
        """:return: List of clauses whose kind is part of `clause_kinds`.
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
        :param bool search_own_clauses: Search the clauses of this directive.
        :param bool search_kernels_region_clauses: Search the clauses of a kernels region
                                                   that embeds this directive.
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
        :note: Assumes number of if clauses of acc data directives has been checked before.
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
        return "", False 

    def transform(self,*args,**kwargs):
        if self.is_purely_declarative():
            return nodes.STNode.transform(self,*args,**kwargs)
        else:
            for dest_dialects, singleton in self.__class__._backends:
                if self.dest_dialect in dest_dialects:
                    singleton.configure(self)
                    return singleton.transform(*args,**kwargs)
        return "", False


class STAccComputeConstruct(STAccDirective, nodes.STComputeConstruct):
    r""" This scanner tree node is encapsulates data and functionality required for
    modelling an offloadable code regions in a Fortran program.

    The scanner component's parser creates a `STAccComputeConstruct` node if a `acc parallel`,
    `acc serial`, `acc parallel loop`, or `acc kernels loop` directive
    is encountered. It is further created if a (potentially) offloadable code
    region was found within an `acc kernels` region.

    :note: Multiple code generation backends can be registered via the `register_backend` class method.
    """
    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, singleton):
        cls._backends.append((dest_dialects, singleton))

    def __init__(self, parent_directive, first_linemap, first_linemap_first_statement):
        STAccDirective.__init__(self, parent_directive, 
                                first_linemap, first_linemap_first_statement)
        nodes.STComputeConstruct.__init__(self, first_linemap,
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
        return "", False

    def get_acc_kernels_ancestor(self):
        """:return: the parent directive if it is an
        `acc kernels` directive or None. """
        if (self.parent_directive != None
           and self.parent_directive.is_directive(["acc","kernels"])):
            return self.parent_directive
        else:
            return None

    def get_acc_data_ancestors(self):
        """:return: all `acc data` ancestors in top-down order."""
        acc_data_directives = []
        current_dir = self
        while current_dir.parent_directive != None:
            current_dir = current_dir.parent_directive    
            if current_dir.is_directive(["acc","data"]):
                acc_data_directives.insert(0,current_dir)
        return acc_data_directives
    
    # overwrite 
    def first_statement(self):
        if self.get_acc_kernels_ancestor() != None:
            return self.parent_directive.first_statement()
        else:
            return STAccDirective.first_statement(self)

    # overwrite 
    def statements(self, include_none_entries=False):
        result = STAccDirective.statements(self, include_none_entries)
        if self.get_acc_kernels_ancestor() != None:
            result.insert(0,self.parent_directive.first_statement())
        return result