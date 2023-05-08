# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing as pyp

def make_str(tokens):
    return "".join([str(tk) for tk in tokens]) 

number = pyp.pyparsing_common.number.copy().setParseAction(make_str)

expr = pyp.Optional(pyp.Group(pyp.delimitedList(number)),default=[])

class ArgumentList():
    def __init__(self,tokens):
        try:
            self.items = tokens[0].asList()
        except AttributeError:
            if isinstance(tokens[0],list):
                self.items = tokens[0]
            else:
                raise
    def __str__(self):
        return str(self.items)
    def __len__(self):
        return len(self.items)
    def __iter__(self):
        return iter(self.items)

expr.setParseAction(ArgumentList)

print(expr.parseString("")[0])
print(expr.parseString("3,5")[0])

myarglist = expr.parseString("3,5")[0]
print(len(myarglist))
for el in myarglist:
    print(el)
