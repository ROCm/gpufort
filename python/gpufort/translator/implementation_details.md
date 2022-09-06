# Translator

This document highlights some of the implementation decisions
when creating the translator class.

## Parser and tree

The parser classifies statements via the `gpufort.util.parsing` package,
and then uses either parsers from that pacakge or a `pyparsing` grammar
to translate statements into a tree representation.
While `pyparsing` subtrees are created via `<pyparsing_grammar_object>.parse_string(<statement_str>)`,
tokens identified via `util.parsing` are translated explicitly into tree nodes via
appropriate constructors.
A `pyparsing`-based translation can be rather slow. This is the reason why we replace more and more pyparsing parse expressions
by dedicated parse routines from the `gpufort.util.parsing` package.
While the OpenACC to OpenMP translation relies on a `pyparsing` grammar for all directives, the OpenACC to HIP conversion
relies only on the directives associated with compute constructs and not with directives that
are translated to OpenACC runtime calls. 

All translator tree nodes are defined in the subpackage `translator.tree`. They all have the prefix the `TT` ("translator tree"); as opposed to the `ST` prefix, which indicates a scanner tree node).

### Arithmetic and logical expressions, complex lvalues and rvalues

Complex lvalue and rvalue expressions such as a Fortran derived type member access
are implemented via `pyparsing.Forward`, which can be used to forward declare certain expressions
and, therefore, to realize a parser for recursive expressions:

**Example 1** (`Forward`, recursive expressions):

```python3
from pyparsing import *
derived_type = Forward()
identifier = pyparsing_common.identifier
derived_type <<= identifier + Literal("%") + (identifier|derived_type)
```

The pyparsing grammar constructed in **Ex. 1** is able to parse
arbitrary recursive expressions such as 

```
a%b1
a%b1%c2_3%d
```

In order to parse arithmetic and logical expressions as they appear in
assignments and conditional expressions as they, e.g., appear in `IF`, `ELSEIF` statements,
we rely on another `pyparsing` shortcut, `infixNotation`, 
which take an rvalue-expresion and a list of operators and their respective number of operands and their
associativity. The position of the operator in the list of operators indicates its preference.

**Example 2** (`infixNotation`):
```python3
import pyparsing as pyp

number     = pyp.pyparsing_common.number.copy().setParseAction(lambda x: str(x))
identifier = pyp.pyparsing_common.identifier
terminal   = identifier | number

expr = pyp.infixNotation(terminal, [
    (pyp.Regex('[\-+]'), 1, pyp.opAssoc.RIGHT),
    (pyp.Regex('[*/+\-]'), 2, pyp.opAssoc.LEFT),
]) + pyp.stringEnd()

print(expr.parseString("-5 + 3"))
print(expr.parseString("-5"))
print(expr.parseString("-( a + (b-1))"))
```


### Loop and compute construct directives

Fortran `DO` (and `DO_CONCURRENT`) loops are the main targets for directive-based offloading moels, which
annotate the former with information on how to map that particular loop (or loopnest) to a device's compute units.
In the GPUFORT translator tree, `DO`-loops have a loop annotation for storing subtrees associated with OpenACC (`acc loop`, `acc parallel loop`, `acc kernels loop`)
and CUDA Fortran directives (`!$cuf kernel do`). These loop annotations implement the interface `translator.tree.directives.ILoopAnnotation` so that
information can be obtained from them in an unified way. A similar interface, `translator.tree.directives.IComputeConstruct`, exists
to represent the different OpenACC compute constructs (`acc kernels`, `acc kernels loop`, `acc parallel loop`, `acc parallel`, `acc serial`)
and the CUDA Fortran construct `!$cuf kernel do` in a unified way.
