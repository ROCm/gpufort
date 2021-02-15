import pyparsing as pp

# grammar
rvalue = pp.pyparsing_common.identifier
op   = pp.Literal("+")

expr = rvalue + op + rvalue

# test
print(expr.parseString("a + b")) # output : ['a','+','b']

# ...
# continue from previous snippet
class RValue():
  def __init__(self,tokens):
    self._value = tokens
class Op():
  def __init__(self,tokens):
    self._op = tokens
rvalue.setParseAction(RValue)
op.setParseAction(Op)

# run test again
print(expr.parseString("a + b")) # output : [<__main__.RValue object ...>, <__main__.Op object ...>, <__main__.RValue object ...>]
