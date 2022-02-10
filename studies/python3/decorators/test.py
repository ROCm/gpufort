#!/usr/bin/env python3
from functools import wraps

def mydecorator(prefix,print_args=False):
    def mydecorator2(func):
        def wrapper(*args,**kwargs):
            if print_args:
                tokens = [func.__name__,",args=",str(args),",kwargs=",str(kwargs)]
            else:
                tokens = []
            print("".join([prefix,":","enter "] + tokens))
            retvals = func(*args,**kwargs)
            print("".join([prefix,":","leave "] + tokens))
        return wrapper
    return mydecorator2

@mydecorator("test")
def foo1():
    print("foo1")

@mydecorator("test")
def foo2(i):
    print("foo2")

@mydecorator("test",True)
def foo3(i,j):
    print("foo3")

foo1()
foo2(5)
foo3(1,2)
