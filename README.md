# gpufort

This project develops a tool for source2source translation of

1. Fortran+OpenACC and CUDA Fortran -> Fortran + OpenMP 4.5+
2. Fortran+OpenACC and CUDA Fortran -> Fortran + [GCC/AOMP OpenACC/MP runtime calls] + HIP C++

The result of the first translation process, can be compiled
with AOMP, which has a Fortran frontend.
The result of the second translation process can be compiled
with hipfort or a combination of hipcc and gfortran.
Note that a OpenACC runtime is only necessary for translating
OpenACC code.

These translation processes are illustrated below:



## Limitations

* `gpufort` is not a compiler (yet)

`gpufort` is not intended to be a compiler.
It's main purpose is to be a translator that allows
an experienced user to fix and tune the outcomes
of the translation process. 
However, we believe `gpufort` can develop into an 
early-outlining compiler if enough effort 
is put into the project.
Given that all code and especially the grammar is
written in python3, `gpufort` can be developed at a quick 
pace.

* `gpufort` does not implement the full OpenACC standard (yet)

`gpufort` was developed to translate a number of HPC apps
to code formats that are well supported by AMD's ROCm ecosystem.
The development of `gpufort` is steered by the requirements
of these applications.

## Outlook

One future goal of the project is that both translation 
processes can be mixed, which will allow users to specify what 
compute directives should be translated to HIP C++ and what compute
directives should be translated to OpenMP.

## Background

## Key ingredient: pyparsing grammars and parse actions

The fundamental ingredient of `gpufort` is its pyparsing grammar that (currently) covers a subset of the Fortran
language that plays a role in computations. This grammar is extended by additional grammar that describes
the structure of CUDA Fortran and OpenACC directives.

While easing development of a parser with shortcuts such as forward declarations and infix notation objects,
pyparsing quickly allows to generate an abstract syntax tree (AST) from a grammar with the aid 
of so-called parse actions.

A simple pyparsing grammar is given below:



```python
import pyparsing as pp

# grammar
rvalue = pp.pyparsing_common.identifier
op   = pp.Literal("+")

expr = rvalue + op + rvalue

# test
print(expr.parseString("a + b")) # output : ['a','+','b']
```

We can directly generate an AST from the parsed string:

```python
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
```

Now instead of the parsed strings, we have three objects in the parse result set
that describe the type of the individual tokens, i.e. we have an abstract syntax tree.
