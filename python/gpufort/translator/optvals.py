# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
class OptionalFlag:
    def __init__(self):
        self.specified = False

class OptionalSingleValue:
    def __init__(self):
        self.specified = False
        self._value = None
    def setvalue(self,value):
        self.specified = True
        self._value = value
    def getvalue(self):
        return self._value
    value = property(getvalue,setvalue) # Expr `obj.value` results in getter/setter call

class OptionalListValue:
    def __init__(self):
        self.value = []
    @property # fake property, expr `obj.specified` results in `obj.specified()` call
    def specified(self):
        return len(self.value) > 0
    def __len__(self):
        return len(self.value)
    def __getitem__(self,key):
        return self.value[key]
    def __iter__(self):
        return iter(self.value)

class OptionalDictValue:
    def __init__(self):
        self.value = {}
    @property
    def specified(self):
        return len(self.value) > 0 
    def __len__(self):
        return len(self.value)
    def __getitem__(self,key):
        return self.value[key]
    def __iter__(self):
        return iter(self.value)
    def items(self):
        return self.value.items()
