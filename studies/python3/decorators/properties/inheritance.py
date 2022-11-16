#!/usr/bin/env python3
class Base:
    def __init__(self):
        self._protected_value = None
        self._base_getter_called = False
        self._base_setter_called = False
    @property
    def value(self):
        print("Base.value")
        self._base_getter_called = True
        return self._protected_value
  
    @value.setter
    def value(self,value):
        print("Base.set_value")
        self._base_setter_called = True
        self._protected_value = value

class Sub(Base):
    pass
 
b = Base()
b.value = 5 # can't set
b.value
print(b._base_getter_called)
print(b._base_setter_called)

s = Sub()
s.value = 5
s.value
print(s._base_getter_called)
print(s._base_setter_called)
