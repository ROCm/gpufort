# ToDos

* [ ] Implement kernel subroutine parser
  * [ ] Remove input parameters from declarations in body; 
    * can be deduced from arglist
    * return type is always void
  * [x] Make grid,dim,sharedMem (int) and stream (type(c_ptr)) 
    arguments to kernel. Do not deduce them

* [ ] Handle struct / derived type arguments to cuf kernels / kernel subroutines
  * ~~Proposed solution: Flatten out the expression, use "TODO ..." as default type~~
  * ~~Eventually need a way to parse derived types ...~~
  * [x] Parse types and demangle all members. Create dummy variables with name "a%b%c"
* [x] Compress linear index mappings
* [x] Identify local variables in cuf kernels  ...
* [x] and remove them from arg list,
      * Scan for scalar lvalues
      * Problem: need also to check if they have been used in function before assignment
        * Proposed solution: 
          * Find all assignments and arithmeticLogicExpression in order
          * Find rvalues and lvalues per expression, i.e have list [ { "lvalue" : [] , "rvalues": [] } ]
          * Check if lvalue was used as rvalue before. If so, do not include in result list
          * Problem: (Minor) When we do adaptive parsing, some lines are commented out, i.e. result might contain errors.
            * Proposed solution: Do ignore result of procedure in this case
* [ ] detect reduced variables
  * Criterion: scalar variable that is assigned but not read anymore
  * [ ] compute partial sum per variable
  * [ ] 
* [ ] Add option to specify search dir as individual files, this allows to filter out some files
      and be more specific what to include in scan and what not.
* [x] More type conversions and conjugate functions (dcmplx, dconjg, ...)
* [ ] Detect "/= 0" expressions and replace with "c_associated(...)" if type is stream or cufft plan (interger) type.

* To replace expression in array evaluations, I need to know the index str / macro from the corresponding declaration.
  * Problem is that I do not know the lower bound. Save lbound as well? 
    * Decision: Ignore for now!

# Code snippets:


Search and replace expression:

```python
for tokens,start,end in expr.scanString(string):
    print(string.replace(string[start:end],"EDIT"))
```

for tokens,start,end in expr.scanString(string):
    tokens[0].configure(extra="50")
    print(string.replace(string[start:end],tokens[0].__str__()))
