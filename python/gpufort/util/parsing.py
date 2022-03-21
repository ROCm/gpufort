# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# TODO use deque

import re

from . import error

COMMENT_CHARS = "!cCdD*"

def tokenize(statement, padded_size=0, modern_fortran=True):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    """
    TOKENS_REMOVE = r"\s+|\t+"
    if modern_fortran:
        TOKENS_KEEP = r"(end|else|![@\$]?|[(),%]|::?|=>?|<<<|>>>|[<>]=?|[\/=]=|\+|-|\*|\/|\.\w+\.)"
    else:
        TOKENS_KEEP = r"(end|else|![@\$]?|^[c\*][@\$]?|[(),%]|::?|=>?|<<<|>>>|[<>]=?|[\/=]=|\+|-|\*|\/|\.\w+\.)"
    # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
    # is not captured.

    tokens1 = re.split(TOKENS_REMOVE, statement)
    tokens = []
    for tk in tokens1:
        tokens += [
            part for part in re.split(TOKENS_KEEP, tk, 0, re.IGNORECASE)
        ]
    result = [tk for tk in tokens if tk != None and len(tk.strip())]
    if padded_size > 0 and len(result) < padded_size:
        return result + [""] * (padded_size - len(result))
    else:
        return result

def mangle_fortran_var_expr(var_expr):
    """Unsophisticated name mangling routine that
    replaces '%(),:' by underscores."""
    result = ""
    tokens = tokenize(var_expr)
    while len(tokens):
        tk = tokens.pop(0)
        if tk == "%":
            result += "_"
        elif tk == ",":
            result += ""
        elif tk == "(":
            result += "L"
        elif tk == ")":
            result += "R"
        elif tk == ":":
            result += "T"
        else:
            result += tk
    return result

def derived_type_parents(var_expr):
    """:return: the derived type parents of a 
    variable if the variable is member of a derived type,
    indicated by the `%` symbol.
    
    :Example: 
    
    Given the expression `A%b(i)%c`, this function
    will return a list ["b","A%b"].
    """
    parts = strip_array_indexing(var_expr).split("%")
    result = []
    current = ""
    for i,_ in enumerate(parts[:-1]): # do not take last part into account
        result.append("%".join(parts[0:i+1]))
    return result

def strip_array_indexing(var_expr):
    """Removes array indexing expressions from variable expressions such as 'A%b(i)%c'.
    The example 'A%b(i)%c' is translated to 'A%b%c'.
    All array indexing expressions are stripped away.

    :param str var_expr: a simple identifier such as 'a' or 'A_d', an array access expression suhc as 'B(i,j)' or a more complicated derived-type member variable expression such as 'a%b%c' or 'A%b(i)%c'.
    :see: indexer.scope.search_index_for_var
    """
    result = var_expr.lstrip("-+") # remove trailing minus sign
    if not "(" in result:
        return result
    else:
        parts = re.split("([()%,])",result)
        open_brackets = 0
        result = []
        curr = ""
        for part in parts:
            if part == "(":
                open_brackets += 1
            elif part == ")":
                open_brackets -= 1
            elif part == "%" and open_brackets == 0:
                result.append(curr)
                curr = ""
            elif open_brackets == 0:
                curr += part
        result.append(curr)
        return "%".join(result)

def split_fortran_line(line):
    global COMMENT_CHARS
    """Decomposes a Fortran line into preceding whitespace,
     statement part, comment part, and trailing whitespace (including newline char).
     """
    len_line = len(line)
    len_preceding_ws = len_line - len(line.lstrip(" \n\t"))
    len_trailing_ws = len_line - len(line.rstrip(" \n\t"))
    content = line[len_preceding_ws:len_line - len_trailing_ws]

    statement = ""
    comment = ""
    if len(content):
        if len_preceding_ws == 0 and content[0] in COMMENT_CHARS or\
           content[0] == "!":
            if len(content) > 1 and content[1] in "$@":
                statement = content
            else:
                comment = content
        else:
            content_parts = content.split("!", maxsplit=1)
            if len(content_parts) > 0:
                statement = content_parts[0]
            if len(content_parts) == 2:
                comment = "!" + content_parts[1]
    preceding_ws = line[0:len_preceding_ws]
    trailing_ws = line[len_line - len_trailing_ws:]
    return preceding_ws, statement.rstrip(" \t"), comment, trailing_ws

def relocate_inline_comments(lines):
    """Move comments that follow after a line continuation char ('&')
    before the 
    Example. Input:
 
    ```
      call myroutine( & ! comment 1
        arg1,&
        ! comment 2


        arg2) ! comment 3
    ```
    Output:
    ```
      call myroutine( & 
        arg1,&
        arg2)
      ! comment 1
      ! comment 2 
      ! comment 3
    ```
    """
    result = []
    comment_buffer = []
    in_multiline_statement = False
    for line in lines:
        indent,stmt_or_dir,comment,trailing_ws = \
          split_fortran_line(line)
        if len(stmt_or_dir) and stmt_or_dir[-1] == "&":
            in_multiline_statement = True
        if in_multiline_statement and len(comment):
            if len(stmt_or_dir):
                result.append("".join([indent, stmt_or_dir, trailing_ws]))
            comment_buffer.append("".join([indent, comment, trailing_ws]))
        else:
            result.append(line)
        if len(stmt_or_dir) and stmt_or_dir[-1] != "&":
            result += comment_buffer
            comment_buffer.clear()
            in_multiline_statement = False
    return result

def next_tokens_till_open_bracket_is_closed(tokens, open_brackets=0, brackets=("(",")"), keepend=True):
    # ex:
    # input:  [  "kind","=","2","*","(","5","+","1",")",")",",","pointer",",","allocatable" ], open_brackets=1
    # result: [  "kind","=","2","*","(","5","+","1",")",")" ]
    result = []
    idx = 0
    criterion = (len(tokens) and tokens[0]==brackets[0]) or open_brackets > 0
    while criterion:
        tk = tokens[idx]
        result.append(tk)
        if tk == brackets[0]:
            open_brackets += 1
        elif tk == brackets[1]:
            open_brackets -= 1
        idx += 1
        criterion = idx < len(tokens) and open_brackets > 0
    # TODO throw error if open_brackets still > 0
    if not keepend:
        result.pop()
    return result

def get_highest_level_arguments(tokens,
                                open_brackets=0,
                                separators=[","],
                                terminators=["::", "\n", "!"],
                                brackets=("(",")")):
    # ex:
    # input: ["parameter",",","intent","(","inout",")",",","dimension","(",":",",",":",")","::"]
    # result : ["parameter", "intent(inout)", "dimension(:,:)" ]
    result = []
    idx = 0
    current_substr = ""
    criterion = len(tokens)
    while criterion:
        tk = tokens[idx]
        idx += 1
        criterion = idx < len(tokens)
        if tk in separators and open_brackets == 0:
            if len(current_substr):
                result.append(current_substr)
            current_substr = ""
        elif tk in terminators:
            criterion = False
        else:
            current_substr += tk
        if tk == brackets[0]:
            open_brackets += 1
        elif tk == brackets[1]:
            open_brackets -= 1
    if len(current_substr):
        result.append(current_substr)
    return result

def extract_function_calls(text, func_name):
    """Extract all calls of the function `func_name` from the input text.
    :param str text: the input text.
    :param str func_name: Name of the function. Can be a regular expression.
    :note: Input text must not contain any line break/continuation characters.
    :return: List of tuples that each contain the function call substring at position 0
             and the list of function call argument substrings at position 1.
    """
    result = []
    for m in re.finditer(r"{}\s*\(".format(func_name), text):
        rest_substr = next_tokens_till_open_bracket_is_closed(
            text[m.end():], open_brackets=1)
        args = get_highest_level_arguments(rest_substr[:-1], 0, [","],
                                           []) # clip the ')' at the end
        end = m.end() + len(rest_substr)
        result.append((text[m.start():end], args))
    return result

def parse_use_statement(statement):
    """Extracts module name and 'only' variables
    from the statement.
    :return: A tuple of containing module name and a list of 'only' 
    tuples (in that order). The 'only' tuples store
    the local name and the module variable name (in that order).
    :raise SyntaxError: If the syntax of the expression is not as expected.

    :Example:
    
    `import mod, only: var1, var2 => var3`

    will result in the tuple:

    ("mod", [("var1","var1"), ("var2","var3)])

    In the example above, `var2` is the local name,
    `var3` is the name of the variable as it appears
    in module 'mod'. In case no '=>' is present,
    the local name is the same as the module name.
    """
    tokens = tokenize(statement)
    only = []
    if len(tokens) > 5 and tokens[2:5] == [",","only",":"]:
        is_local_name = True
        for tk in tokens[5:]:
            if tk == ",":
                is_local_name = True
            elif tk == "=>":
                is_local_name = False
            elif tk.isidentifier() and is_local_name:
                only.append((tk,tk))
            elif tk.isidentifier() and not is_local_name:
                local_name,_ = only[-1]
                only[-1] = (local_name, tk)
            else:
                raise error.SyntaxError("could not parse 'use' statement")
    elif len(tokens) != 2:
        raise error.SyntaxError("could not parse 'use' statement")
    return tokens[1], only

def parse_attributes_statement(statement):
    """Parses a CUDA Fortran attributes statement.
    :return: A tuple with the attribute as first
            and the subjected variables (list of str)
            as second entry.
   
    :Example:
   
    `attributes(device) :: var_a, var_b`
   
    will result in 

    ('device', ['var_a', 'var_b'])
    """
    tokens = tokenize(statement,10)

    attributes = []
    if tokens.pop(0)!="attributes":
        raise error.SyntaxError("expected 'attributes' keyword")
    if tokens.pop(0)!="(":
        raise error.SyntaxError("expected '('")
    i = 0
    tk = tokens.pop(0)
    condition = True
    while ( condition ):
        if i % 2 == 0:
           if tk.isidentifier():
               attributes.append(tk)
           else:
               raise error.SyntaxError("expected identifier")
        elif tk not in [","]:
            raise error.SyntaxError("expected ','")
        i += 1
        tk = tokens.pop(0)
        condition = tk != ")"
    if tk!=")":
        raise error.SyntaxError("expected ')'")
    if tokens.pop(0)!="::":
        raise error.SyntaxError("expected '::'")
    # variables
    variables = []
    i = 0
    for i,tk in enumerate(tokens):
        if len(tk):
            if i % 2 == 0:
               if tk.isidentifier():
                   variables.append(tk)
               else:
                   raise error.SyntaxError("expected identifier")
            elif tk != ",":
                raise error.SyntaxError("expected ','")
    return attributes, variables    

def parse_declaration(statement):
    """Decomposes a Fortran declaration into its individual
    parts.

    :return: A tuple (datatype, kind, qualifiers, dimension_bounds, var_list, qualifiers_raw)
             where `qualifiers` contain qualifiers except the `dimension` qualifier,
             `dimension_bounds` are the arguments to the `dimension` qualifier,
             var_list consists of triples (varname, bounds or [], right-hand-side or None),
             and `qualifiers_raw` contains the full list of qualifiers (without whitespaces).
    :note: case-ins
    :raise error.SyntaxError: If the syntax of the expression is not as expected.
    """
    tokens = tokenize(statement,10)

    idx_last_consumed_token = None
    # handle datatype
    datatype = tokens[0]
    kind = None
    DOUBLE_COLON = "::"
    try:
        tokens_lower = list(map(str.lower, tokens))
        if tokens_lower[0] == "type":
            # ex: type ( dim3 )
            kind = tokens[2]
            idx_last_consumed_token = 3
        elif tokens_lower[0:2] == ["double", "precision"]:
            datatype = " ".join(tokens[0:2])
            idx_last_consumed_token = 1
        elif tokens[1] == "*":
            # ex: integer * 4
            # ex: integer * ( 4*2 )
            if tokens[2] == "(":
                kind_tokens = next_tokens_till_open_bracket_is_closed(
                    tokens[3:], open_brackets=1)
            else:
                kind_tokens = tokens[2] 
            kind = "".join(kind_tokens)
            idx_last_consumed_token = 2 + len(kind_tokens) - 1
        elif tokens_lower[1:4] == ["(","kind","("]:
            # ex: integer ( kind ( 4 ) )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[4:], open_brackets=1)
            kind = "".join(tokens[2:4]+kind_tokens)
            idx_last_consumed_token = 4 + len(kind_tokens)
        elif (tokens_lower[1:4] == ["(","kind","="] 
             or tokens_lower[1:4] == ["(","len","="]):
            # ex: integer ( kind = 4 )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[4:], open_brackets=1)
            kind = "".join(kind_tokens[:-1])
            idx_last_consumed_token = 4 + len(kind_tokens)
        elif tokens[1] == "(":
            # ex: integer ( 4 )
            # ex: integer ( 4*2 )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[2:], open_brackets=1)
            kind = "".join(kind_tokens[:-1])
            idx_last_consumed_token = 2 + len(kind_tokens) - 1
        elif tokens[1] in [",", DOUBLE_COLON] or tokens[1].isidentifier():
            # ex: integer ,
            # ex: integer ::
            # ex: integer a
            idx_last_consumed_token = 0
        else:
            raise error.SyntaxError("could not parse datatype")
    except IndexError:
        raise error.SyntaxError("could not parse datatype")
    # handle qualifiers
    try:
        datatype_raw = "".join(tokens[:idx_last_consumed_token+1])
        tokens = tokens[idx_last_consumed_token + 1:] # remove type part tokens
        idx_last_consumed_token = None
        if tokens[0] == "," and DOUBLE_COLON in tokens:
            qualifiers = get_highest_level_arguments(tokens)
            idx_last_consumed_token = tokens.index(DOUBLE_COLON)
        elif tokens[0] == DOUBLE_COLON:
            idx_last_consumed_token = 0
        elif tokens[0] == ",":
            raise error.SyntaxError("could not parse qualifier list")
        qualifiers_raw = get_highest_level_arguments(tokens)

        # handle variables list
        if idx_last_consumed_token != None:
            tokens = tokens[idx_last_consumed_token+1:] # remove qualifier list tokens
        variables_raw = get_highest_level_arguments(tokens)
    except IndexError:
        raise error.SyntaxError("could not parse qualifier list")

    # construct declaration tree node
    qualifiers = []
    dimension_bounds = []
    for qualifier in qualifiers_raw:
        qualifier_tokens       = tokenize(qualifier)
        qualifier_tokens_lower = list(map(str.lower, qualifier_tokens))
        if qualifier_tokens_lower[0] == "dimension":
            # ex: dimension ( 1, 2, 1:n )
            qualifier_tokens = tokenize(qualifier)
            if (len(qualifier_tokens) < 4
               or qualifier_tokens_lower[0:2] != ["dimension","("]
               or qualifier_tokens[-1] != ")"):
                raise error.SyntaxError("could not parse 'dimension' qualifier")
            dimension_bounds = get_highest_level_arguments(qualifier_tokens[2:-1])
        elif qualifier_tokens_lower[0] == "intent":
            # ex: intent ( inout )
            qualifier_tokens = tokenize(qualifier)
            if (len(qualifier_tokens) != 4 
               or qualifier_tokens_lower[0:2] != ["intent","("]
               or qualifier_tokens[-1] != ")"
               or qualifier_tokens_lower[2] not in ["out","inout","in"]):
                raise error.SyntaxError("could not parse 'intent' qualifier")
            qualifiers.append("".join(qualifier_tokens))
        elif len(qualifier_tokens)==1 and qualifier_tokens[0].isidentifier():
            qualifiers.append(qualifier)
        else:
            raise error.SyntaxError("could not parse qualifier '{}'".format(qualifier))
        
    variables = []
    # 
    for var in variables_raw:
        var_tokens = tokenize(var)
        var_name   = var_tokens.pop(0)
        var_bounds = [] 
        var_rhs    = None
        # handle bounds
        if (len(var_tokens) > 2
           and var_tokens[0] == "("):
            if len(dimension_bounds):
                raise error.SyntaxError("'dimension' and variable cannot be specified both")
            bounds_tokens = next_tokens_till_open_bracket_is_closed(var_tokens)
            var_bounds = get_highest_level_arguments(bounds_tokens[1:-1])
            for i in range(0,len(bounds_tokens)):
                var_tokens.pop(0)
        if (len(var_tokens) > 1 and
           var_tokens[0] in ["=>","="]):
            var_tokens.pop(0)
            var_rhs = "".join(var_tokens)
        elif len(var_tokens):
            raise error.SyntaxError("could not parse '{}'".format(var_tokens))
        variables.append((var_name,var_bounds,var_rhs))
    return (datatype, kind, qualifiers, dimension_bounds, variables, datatype_raw, qualifiers_raw) 

def parse_directive(directive,tokens_to_omit_at_begin=0):
    """Splits a directive in directive sentinel, 
    identifiers and clause-like expressions.

    :return: List of directive parts as strings.
    :param int tokens_to_omit_at_begin: Number of tokens to omit at the begin of the result.
                                        Can be used to get only the tokens that
                                        belong to the clauses of the directive if 
                                        the directive kind is already known.

    :Example:

    Given the directive "$!acc data copyin ( a (1,:),b,c) async",
    this routine will return a list (tokens_to_omit_at_begin = 0)
    ['$!','acc','data','copyin(a(1,:),b,c)','async']
    If tokens_to_omit_at_begin = 3, this routine will return
    ['copyin(a(1,:),b,c)','async']
    """
    tokens = tokenize(directive)
    it = 0
    result = []
    while len(tokens):
        tk = tokens.pop(0) 
        if tk == "(":
            rest = next_tokens_till_open_bracket_is_closed(tokens,1)
            result[-1] = "".join([result[-1],tk]+rest)
            for i in range(0,len(rest)):
                tokens.pop(0)
        elif tk.isidentifier() or tk.endswith("$"):
            result.append(tk)
        else:
            raise error.SyntaxError("unexpected clause or identifier token: {}".format(tk))
    return result[tokens_to_omit_at_begin:]

def parse_acc_directive(directive):
    """Parses an OpenACC directive, returns a triple of its sentinel, its kind (as list of tokens)
    plus its clauses (as list of tokens). 

    :Example:

    Given
      `!$acc enter data copyin(a,b) async`,
    this routine returns
      ('!$',['acc','enter','data'],['copyin(a,b)','async'])
    """
    directive_parts = parse_directive(directive)
    directive_kinds = [
      ["acc","init"],
      ["acc","shutdown"],
      ["acc","data"],
      ["acc","enter","data"],
      ["acc","exit","data"],
      ["acc","wait"],
      ["acc","update"],
      ["acc","declare"],
      ["acc","routine"],
      ["acc","loop"],
      ["acc","parallel","loop"],
      ["acc","kernels","loop"],
      ["acc","parallel"],
      ["acc","kernels"],
      ["acc","serial"],
      ["acc","end","data"],
      ["acc","end","loop"],
      ["acc","end","parallel","loop"],
      ["acc","end","kernels","loop"],
      ["acc","end","parallel"],
      ["acc","end","kernels"],
      ["acc","end","serial"],
    ]
    directive_kind = []
    directive_args = []
    clauses        = []
    sentinel = directive_parts.pop(0) # get rid of sentinel
    for kind in directive_kinds:
        length = len(kind)
        if len(directive_parts) >= length:
            last_directive_token_parts = directive_parts[length-1].split("(")
            if (directive_parts[0:length-1]+last_directive_token_parts[0:1])==kind:
                directive_kind = kind
                if len(last_directive_token_parts) > 1:
                    _, directive_args = parse_acc_clauses([directive_parts[length-1]])[0] 
                clauses = directive_parts[length:]
                break
    if not len(directive_kind):
        raise error.SyntaxError("could not identify kind of directive")
    return sentinel, directive_kind, directive_args, clauses

def parse_acc_clauses(clauses):
    """Converts OpenACC-like clauses into tuples with the name
    of the clause as first argument and the list of arguments of the clause
    as second argument, might be empty.
    :param list clauses: A list of acc clause expressions (strings).
    """
    result = []
    for expr in clauses:
        tokens = tokenize(expr)
        if len(tokens) == 1 and tokens[0].isidentifier():
            result.append((tokens[0],[]))
        elif (len(tokens) > 3 
             and tokens[0].isidentifier()
             and tokens[1] == "(" 
             and tokens[-1] == ")"):
            args1 = get_highest_level_arguments(tokens[2:-1], 0, [","],[]) # clip the ')' at the end
            args = []
            current_group = None
            for arg in args1:
                arg_parts = tokenize(arg)
                if len(arg_parts)>1 and arg_parts[1]==":":
                    if current_group != None:
                        args.append(current_group)
                    current_group = ( arg_parts[0], ["".join(arg_parts[2:])] )
                elif current_group != None:
                    current_group[1].append(arg)
                else:
                    args.append(arg)
            if current_group != None:
                args.append(current_group)
            result.append((tokens[0], args))
        else:
            raise error.SyntaxError("could not parse expression: '{}'".format(expr))
    return result

def parse_cuf_kernel_call(statement):
    """Parses a CUDA Fortran kernel call statement.
    Decomposes it into kernel name, launch parameters,
    and kernel arguments.
    
    :Example:

    `call mykernel<<grid,dim>>(arg0,arg1,arg2(1:n))`

    will result in a triple

    ("mykernel",["grid","dim"],["arg0","arg1","arg2(1:n)"])
    """
    tokens = tokenize(statement)
    if tokens.pop(0).lower() != "call":
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'call'")
    kernel_name = tokens.pop(0)
    # parameters
    if not kernel_name.isidentifier():
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'identifier'")
    if tokens.pop(0) != "<<<":
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected '<<<'")
    params_tokens = next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, brackets=("<<<",">>>"), keepend=False)
    for i in range(0,len(params_tokens)):
        tokens.pop(0)
    if tokens.pop(0) != ">>>":
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected '>>>'")
    # arguments
    if tokens.pop(0) != "(":
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected '('")
    args_tokens = next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, keepend=False)
    for i in range(0,len(args_tokens)):
        tokens.pop(0)
    if tokens.pop(0) != ")":
        return error.SyntaxError("could not parse CUDA Fortran kernel call: expected ')'")
    params = get_highest_level_arguments(params_tokens,terminators=[])
    args = get_highest_level_arguments(args_tokens,terminators=[])
    if len(tokens):
        return error.SyntaxError("could not parse CUDA Fortran kernel call: trailing text")
    return kernel_name, params, args

# def parse_implicit_statement(statement):
# TODO
#     pass
# def parse_dimension_statement(statement):
# TODO
#     pass
#def parse_allocate_statement(statement):
#    """Parses:
#  
#      (deallocate|allocate) (allocation-list [, stat=stat-variable])
#
#    where each entry in allocation-list has the following syntax:
#
#      name( [lower-bound :] upper-bound [, ...] )  
#
#    :return: List of tuples pairing variable names and bound information plus
#             a status variable expression if the `stat` variable is set. Otherwise
#             the second return value is None.
#
#    :Example:
#    
#    `allocate(a(1,N),b(-1:m,n)`
#
#    will result in the list:
#
#    [("a",[("1","N")]),("b",[("-1","m"),("1","n)])]
#
#    where `1` is the default lower bound if none is present.
#    """
#    result1 = [] 
#    result2 = None
#    args = get_highest_level_arguments(tokenize(text)[2:-1]) # : [...,"b(-1:m,n)"]
#    for arg in args:
#        tokens = tokenize(arg)
#        if tokens[0].lower() == "stat":
#            result2 = " ".join(tokens[2:])
#        else:
#            varname = tokens[0] # : "b"
#            bounds = get_highest_level_arguments(tokens[2:]) # : ["-1:m", "n"]
#            for bound in bounds:
#                bound.split(":")
#            pair = (tokens[0],[])
#    return result1, result2

# rules
def is_declaration(tokens):
    return\
        tokens[0] in ["type","integer","real","complex","logical","character"] or\
        tokens[0:1+1] == ["double","precision"]


def is_ignored_statement(tokens):
    """All statements beginning with the tokens below are ignored.
    """
    return tokens[0] in ["write","print","character","use","implicit"] or\
           is_declaration(tokens)


def is_blank_line(statement):
    return not len(statement.strip())


def is_fortran_directive(statement,modern_fortran):
    """If the statement is a directive."""
    return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:2] == "!$" 
       or (not modern_fortran and statement[0:2] in ["c$","*$"]))

def is_cuda_fortran_conditional_code(statement,modern_fortran):
    """If the statement is a directive."""
    return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:5] == "!@cuf" 
       or (not modern_fortran and statement[0:5] in ["c@cuf","*@cuf"]))

def is_fortran_comment(statement,modern_fortran):
    """If the statement is a directive.
    :note: Assumes that `is_cuda_fortran_conditional_code` and 
           `is_fortran_directive` are checked before.
    """
    return len(statement) and (modern_fortran and statement.lstrip()[0] == "!" 
       or (not modern_fortran and statement[0] in "c*"))


def is_cpp_directive(statement):
    return statement[0] == "#"

def is_ignored_fortran_directive(tokens):
    return tokens[1:2+1] == ["acc","end"] and\
           tokens[3] in ["kernels","parallel","loop"]


def is_fortran_offload_region_directive(tokens):
    return\
        tokens[1:2+1] == ["acc","parallel"] or\
        tokens[1:2+1] == ["acc","kernels"]


def is_fortran_offload_region_plus_loop_directive(tokens):
    return\
        tokens[1:3+1] == ["cuf","kernel","do"]  or\
        tokens[1:3+1] == ["acc","parallel","loop"] or\
        tokens[1:3+1] == ["acc","kernels","loop"] or\
        tokens[1:3+1] == ["acc","kernels","loop"]


def is_fortran_offload_loop_directive(tokens):
    return\
        tokens[1:2+1] == ["acc","loop"]


def is_assignment(tokens):
    #assert not is_declaration_(tokens)
    #assert not is_do_(tokens)
    return "=" in tokens


def is_pointer_assignment(tokens):
    #assert not is_ignored_statement_(tokens)
    #assert not is_declaration_(tokens)
    return "=>" in tokens


def is_subroutine_call(tokens):
    return tokens[0] == "call" and tokens[1].isidentifier()


def is_select_case(tokens):
    cond1 = tokens[0:1 + 1] == ["select", "case"]
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2:3+1] == ["select","case"]
    return cond1 or cond2


def is_case(tokens):
    return tokens[0:1 + 1] == ["case", "("]


def is_case_default(tokens):
    return tokens[0:1 + 1] == ["case", "default"]


def is_if_then(tokens):
    """:note: we assume single-line if have been
    transformed in preprocessing step."""
    return tokens[0:1 + 1] == ["if", "("]


def is_if_then(tokens):
    """:note: we assume single-line if have been
    transformed in preprocessing step."""
    return tokens[0:1 + 1] == ["if", "("]


def is_else_if_then(tokens):
    return tokens[0:2 + 1] == ["else", "if", "("]


def is_else(tokens):
    #assert not is_else_if_then_(tokens)
    return tokens[0] == "else"


def is_do_while(tokens):
    cond1 = tokens[0:1 + 1] == ["do", "while"]
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2:3+1] == ["do","while"]
    return cond1 or cond2


def is_do(tokens):
    cond1 = tokens[0] == "do"
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2] == "do"
    return cond1 or cond2


def is_end(tokens, kinds=[]):
    cond1 = tokens[0] == "end"
    cond2 = not len(kinds) or tokens[1] in kinds
    return cond1 and cond2
