#!/usr/bin/env python3
import addtoplevelpath
import sys
import test
import grammar as grammar

print(grammar.use.parseString("use kinds, only: dp, sp => sp2"))
