# Setting

* GFORTRAN 9.4.0 has been used for all Fortran tests.
* G++ 9.4.0 has been used for all C++ tests.

# Conclusions

* Fortran: CONTINUE, CYCLE, EXIT behave the same way in 
  nested archaic Fortran DO loops with differing and shared label,
  i.e. only the innermost loop is affected by these statements.

* C++ vs Fortran: CONTINUE is equivalent to an empty statement (";"),
  CYCLE can be translated to "continue;" and EXIT to "break;"

## HIP kernel generation

* A CYCLE statement within a collapsed loop must be replaced by a RETURN
  statement ("return;" in C++) in the kernel body associated with a collapsed loop.

* A EXIT statement within a collapsed loop must be replaced by a RETURN
  statement ("return;" in C++) in the kernel body associated with a collapsed loop.

* CONTINUE statements within the loop have relevance for GOTO statements
  and DO loops. They must be replaced by the associated label (if any) and a
  empty statement (";").
