# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from ... import opts

from .. import nodes

class STAccDirective(nodes.STDirective):
    _backends = []

    @classmethod
    def register_backend(cls, dest_dialects, singleton):
        cls._backends.append((dest_dialects, singleton))

    """Class for handling ACC directives."""

    def __init__(self, first_linemap, first_linemap_first_statement,
                 directive_no):
        nodes.STDirective.__init__(self,
                                   first_linemap,
                                   first_linemap_first_statement,
                                   directive_no,
                                   sentinel="!$acc")
        self.tokens = util.parsing.tokenize(self.first_statement())
        self._default_present_vars = []
        self.dest_dialect = opts.destination_dialect

    def is_directive(self,kind=[]):
        """:return if this is a directive of the given kind,
                   where kind is a list of tokens such as 
                   ['acc','enter','region'].
        """
        return self.tokens[1:1+len(kind)] == kind 

    def is_purely_declarative(self):
        return (self.is_directive(["acc","declare"])
               or self.is_directive(["acc","routine"]))
 
    def __str__(self):
        result = []
        for kind in [ 
            ["acc","init"],
            ["acc","shutdown"],
            ["acc","end"],
            ["acc","enter","data"],
            ["acc","exit","data"],
            ["acc","wait"],
            ["acc","loop"],
            ["acc","parallel"],
            ["acc","kernels"],
            ["acc","parallel","loop"],
            ["acc","kernels","loop"],
            ]:
            result.append("-".join(kind)+":"+str(self.is_directive(kind)))
        return ",".join(result)
    __repr__ = __str__

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
    def transform(self,*args,**kwargs):
        for dest_dialects, singleton in self.__class__._backends:
            if self.dest_dialect in dest_dialects:
                singleton.configure(self)
                return singleton.transform(*args,**kwargs)
