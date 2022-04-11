# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# TODO use deque

import re

from . import error

COMMENT_CHARS = "!cCdD*"

def compare_tokens_ignore_case(tokens1,tokens2):
    if len(tokens1) != len(tokens2):
        return False
    else:
        for i,tk in enumerate(tokens1):
            if tk.lower() != tokens2[i].lower():
                return False
        return True

def tokenize(statement, padded_size=0, modern_fortran=True,keepws=False):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    :param bool keepws: keep whitespaces in the resulting list of tokens.
    """
    TOKENS_KEEP = [
        r"[\"](?:\\.|[^\"\\])*[\"]",
        r"[\'](?:\\.|[^\'\\])*[\']",
        r"\bend(?:if|do)?\b",
        r"\belse(?:if)?\b",
        r"[(),%]",
        r"::?",
        r"=>?",
        r"<<<",
        r">>>",
        r"[<>]=?",
        r"[\/=]=",
        r"\+",
        r"-",
        r"\*",
        r"\/",
        r"\.\w+\.",
        r"\s+|\t+",
    ]
    if modern_fortran:
        TOKENS_KEEP.append(r"![@\$]?")
    else:
        TOKENS_KEEP.append(r"^[c\*][@\$]?")
    # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
    # is not captured.
    keep_pattern = "".join(["(","|".join(TOKENS_KEEP),")"])
    
    tokens1 = re.split(keep_pattern, statement, 0, re.IGNORECASE)
    tokens = []
    for tk in tokens1:
        if tk.lower() in ["endif", "elseif", "enddo"]:
            tokens.append(tk[:-2])
            tokens.append(tk[-2:])
        else:
            tokens.append(tk)
    result = [tk for tk in tokens if tk != None and (keepws or len(tk.strip()))]
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

def get_top_level_operands(tokens,
                           separators=[","],
                           terminators=["::", "\n", "!",""],
                           brackets=("(",")"),
                           keep_separators=False):
    """
    :return: Tuple of highest level arguments and number of consumed tokens
             in that order.
    """
    # ex:
    # input: ["parameter",",","intent","(","inout",")",",","dimension","(",":",",",":",")","::"]
    # result : ["parameter", "intent(inout)", "dimension(:,:)" ], i.e. terminators are excluded
    result = []
    idx = 0
    current_substr = ""
    criterion = len(tokens)
    open_brackets = 0
    num_consumed_tokens = 0
    while criterion:
        tk = tokens[idx]
        tk_lower = tk.lower()
        idx += 1
        criterion = idx < len(tokens)
        #
        if tk == brackets[0]:
            open_brackets += 1
        elif tk == brackets[1]:
            open_brackets -= 1
        #
        num_consumed_tokens += 1
        if tk_lower in separators and open_brackets == 0:
            if len(current_substr):
                result.append(current_substr)
            if keep_separators:
                result.append(tk)
            current_substr = ""
        elif tk_lower in terminators or open_brackets < 0:
            criterion = False
            num_consumed_tokens -= 1
        else: 
            current_substr += tk
    if len(current_substr):
        result.append(current_substr)
    return result, num_consumed_tokens

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
        args,_  = get_top_level_operands(rest_substr[:], [","],
                                           [])
        end = m.end() + len(rest_substr)
        result.append((text[m.start():end], args))
    return result

def parse_use_statement(statement):
    """Extracts module name, qualifiers, renamings and 'only' mappings
    from the statement.
    :return: A 4-tuple containing module name, qualifiers, list of renaming tuples, and a list of 'only' 
    tuples (in that order). The renaming and 'only' tuples store
    the local name and the module variable name (in that order).
    :raise SyntaxError: If the syntax of the expression is not as expected.

    :Examples:
    
    `import mod, only: var1, var2 => var3`
    `import, intrinsic :: mod, only: var1, var2 => var3`

    will result in the tuple:

    ("mod", [], [("var1","var1"), ("var2","var3)])
    ("mod", ['intrinsic'], [("var1","var1"), ("var2","var3)])

    In the example above, `var2` is the local name,
    `var3` is the name of the variable as it appears
    in module 'mod'. In case no '=>' is present,
    the local name is the same as the module name.
    """
    tokens = tokenize(statement,10)
    if tokens.pop(0).lower() != "use":
        raise error.SyntaxError("expected 'use'")
    module_name = None
    qualifiers  = []
    tk = tokens.pop(0)
    if tk.isidentifier():
        module_name = tk
    elif tk == ",":
        qualifiers,num_consumed_tokens = get_top_level_operands(tokens)
        for i in range(0,num_consumed_tokens):
            tokens.pop(0)
        if tokens.pop(0) != "::":
            raise error.SyntaxError("expected '::' after qualifier list")
        tk = tokens.pop(0)
        if tk.isidentifier():
            module_name = tk
    else: 
        raise error.SyntaxError("expected ',' or identifier after 'use'")
    mappings = []
    have_only      = False
    have_renamings = False
    if (len(tokens) > 3 
       and tokens[0] == ","
       and tokens[1].lower() == "only"
       and tokens[2] == ":"):
        have_only = True
    elif (len(tokens) > 3 
         and tokens[0] == ","):
        have_renamings = True
    if have_only:    
        for i in range(0,3):
            tokens.pop(0)
    if have_only or have_renamings:
        is_local_name = True
        while len(tokens):
            tk = tokens.pop(0)
            if tk == ",":
                is_local_name = True
            elif tk == "=>":
                is_local_name = False
            elif tk.isidentifier() and is_local_name:
                mappings.append((tk,tk))
            elif tk.isidentifier() and not is_local_name:
                local_name,_ = mappings[-1]
                mappings[-1] = (local_name, tk)
            elif tk != "":
                raise error.SyntaxError("could not parse 'use' statement. Unexpected token '{}'".format(tk))
    if have_only:
        only = mappings
        renamings = []
    else:
        only = []
        renamings = mappings
    if len(tokens):
        while len(tokens) and tokens.pop(0) == "": pass
    if len(tokens):
        raise error.SyntaxError("expected end of statement or ', only:' or ', <expr1> => <expr2>' after module name")
    return module_name, qualifiers, renamings, only

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
                    tokens[3:], open_brackets=1)[:-1]
                kind = "".join(kind_tokens)
                idx_last_consumed_token = 2 + len(kind_tokens) + 1 # )
            else:
                kind = tokens[2] 
                idx_last_consumed_token = 2
        elif tokens_lower[1:4] == ["(","kind","("]:
            # ex: integer ( kind ( 4 ) )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[4:], open_brackets=2)
            kind = "".join(tokens[2:4]+kind_tokens[:-1])
            idx_last_consumed_token = 4 + len(kind_tokens) - 1
        elif (tokens_lower[1:4] == ["(","kind","="] 
             or tokens_lower[1:4] == ["(","len","="]):
            # ex: integer ( kind = 4 )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[4:], open_brackets=1)
            kind = "".join(kind_tokens[:-1])
            idx_last_consumed_token = 4 + len(kind_tokens) - 1
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
    qualifiers_raw = []
    try:
        datatype_raw = "".join(tokens[:idx_last_consumed_token+1])
        tokens = tokens[idx_last_consumed_token + 1:] # remove type part tokens
        idx_last_consumed_token = None
        if tokens[0] == "," and DOUBLE_COLON in tokens: # qualifier list
            qualifiers_raw,_  = get_top_level_operands(tokens)
            idx_last_consumed_token = tokens.index(DOUBLE_COLON)
        elif tokens[0] == DOUBLE_COLON: # variables begin afterwards
            idx_last_consumed_token = 0
        elif tokens[0].isidentifier(): # variables begin directly
            idx_last_consumed_token = -1
        else:
            raise error.SyntaxError("could not parse qualifier list")
        # handle variables list
        tokens = tokens[idx_last_consumed_token+1:] # remove qualifier list tokens
        variables_raw,_  = get_top_level_operands(tokens)
    except IndexError:
        raise error.SyntaxError("could not parse qualifier list")
    # analyze qualifiers
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
            dimension_bounds,_  = get_top_level_operands(qualifier_tokens[2:-1])
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
    # analyze variables
    variables = []
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
            var_bounds,_  = get_top_level_operands(bounds_tokens[1:-1])
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

def parse_derived_type_statement(statement):
    """Parse the first statement of derived type declaration.
    :return: A triple consisting of the type name, its attributes, and
             its parameters (in that order).    
    :Example:
  
    Given the statements

    'type mytype',
    'type :: mytype',
    'type, bind(c) :: mytype',
    'type :: mytype(k,l)'
    
    this routine will return

    ('mytype', [], []),
    ('mytype', [], []),
    ('mytype', ['bind(c)'], []),
    ('mytype', [], ['k','l'])
    """
    tokens = tokenize(statement,padded_size=10)
    if not tokens.pop(0).lower() == "type":
        raise error.SyntaxError("expected 'type'")
    name       = None
    tk = tokens.pop(0)
    parse_attributes = False
    have_name = False
    if tk.isidentifier():
        name = tk 
        have_name = True
        # do parameters
        pass
    elif tk == "::":
        # do name
        # do parameters
        pass
    elif tk == ",":
        # do qualifiers
        # do parameters
        parse_attributes = True
    else:
        raise error.SyntaxError("expected ',','::', or identifier")
    parameters = []
    if parse_attributes:
        attributes,_  = get_top_level_operands(tokens)
        if not len(attributes):
            raise error.SyntaxError("expected at least one derived type attribute")
        while tokens.pop(0) != "::":
            pass
    else:
        attributes = []
    tk = tokens.pop(0)
    if not have_name and tk.isidentifier():
        name = tk 
    elif not have_name:
        raise error.SyntaxError("expected identifier")
    tk = tokens.pop(0)
    if tk == "(": # qualifier list
        if not len(tokens):
            raise error.SyntaxError("expected list of parameters")
        parameters,_  = get_top_level_operands(tokens[:-1])
        if not len(parameters):
            raise error.SyntaxError("expected at least one derived type parameter")
        tk = tokens.pop(0)
        while tk not in ["",")"]:
            tk = tokens.pop(0)
            pass
        if not tk == ")":
            raise error.SyntaxError("expected ')'")
    elif not tk == "":
        raise error.SyntaxError("expected ')'")
    return name, attributes, parameters

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
      ["acc","host_data"],
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
      ["acc","end","host_data"],
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
            args1,_  = get_top_level_operands(tokens[2:-1], [","],[]) # clip the ')' at the end
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
    params,_  = get_top_level_operands(params_tokens,terminators=[])
    args,_  = get_top_level_operands(args_tokens,terminators=[])
    if len(tokens):
        return error.SyntaxError("could not parse CUDA Fortran kernel call: trailing text")
    return kernel_name, params, args

# def parse_implicit_statement(statement):
# TODO
#     pass
# def parse_dimension_statement(statement):
# TODO
#     pass

def parse_do_statement(statement):
    """ High-level parser for `[label:] do var = lbound, ubound [,stride]`
    where `label`, `var` are identifiers and `lbound`, `ubound`, and `stride` are
    arithmetic expressions.

    :return: Tuple consisting of label, var, lbound, ubound, stride (in that order).
             label and stride are None if these optional parts could not be found.

    :Example:

    `mylabel: do i = 1,N+m`

    results in

    ('mylabel', 'i', '1', 'N+m', None)
    """
    tokens = tokenize(statement)
    label = None
    if tokens[0].lower() == "do":
        tokens.pop(0)
    elif (tokens[0].isidentifier()
         and tokens[1] == ":"
         and tokens[2].lower() == "do"):
        label = tokens.pop(0)
        for i in range(0,2): tokens.pop(0)
    else:
        raise error.SyntaxError("expected 'do' or identifier + ':' + 'do'")
    #
    var = tokens.pop(0) 
    if not var.isidentifier():
        raise error.SyntaxError("expected identifier")
    if not tokens.pop(0) == "=":
        raise error.SyntaxError("expected '='")
    range_vals,consumed_tokens  = get_top_level_operands(tokens) 
    lbound = None
    ubound = None
    stride = None
    if len(range_vals) < 2:
        raise error.SyntaxError("invalid loop range: expected at least first and last loop index")
    elif len(range_vals) > 3:
        raise error.SyntaxError("invalid loop range: expected not more than three values (first index, last index, stride)")
    elif len(range_vals) == 2:
        lbound = range_vals[0]
        ubound = range_vals[1]
    elif len(range_vals) == 3:
        lbound = range_vals[0]
        ubound = range_vals[1]
        stride = None
    return (label, var, lbound, ubound, stride)

def parse_deallocate_statement(statement):
    """ Parses `deallocate ( var-list [, stat=stat-variable] )`
    
    where var-list is a comma-separated list of pointers or allocatable
    arrays.
   
    :return: Tuple consisting of list of variable names plus a status variable expression.
             The latter is None if no status var expression was detected.

    :Example:
    
    `allocate(a(1:N),b(-1:m:2,n))`

    will result in the tuple:

    (['a','b'], 'ierr')
    """
    result1 = [] 
    result2 = None
    tokens = tokenize(statement)
    if not tokens.pop(0).lower() == "deallocate":
        raise error.SyntaxError("expected 'deallocate'")
    if tokens.pop(0) != "(":
        raise error.SyntaxError("expected '('")
    args,_  = get_top_level_operands(tokens) # : [...,"b(-1:m,n)"]
    for arg in args:
        parts,_ = get_top_level_operands(tokenize(arg),separators=["%"])
        child_tokens = tokenize(parts[-1])
        if child_tokens[0].lower() == "stat":
            if result2 != None:
                raise error.SyntaxError("expected only a single 'stat' expression")
            child_tokens.pop(0)
            if child_tokens.pop(0) != "=":
                raise error.SyntaxError("expected '='")
            result2 = "".join(child_tokens)
        else:
            varname = child_tokens.pop(0) # : "b"
            if not varname.isidentifier():
                raise error.SyntaxError("expected identifier")
            if len(parts) > 1:
                varname = "%".join(parts[:-1]+[varname])
            result1.append(varname)
    return result1, result2

def parse_allocate_statement(statement):
    """Parses `allocate ( allocation-list [, stat=stat-variable] )`

    where each entry in allocation-list has the following syntax:

      name( [lower-bound :] upper-bound [:stride] [, ...] )  

    :return: List of tuples pairing variable names and bound information plus
             a status variable expression if the `stat` variable is set. Otherwise
             the second return value is None.

    :Example:
    
    `allocate(a(1:N),b(-1:m:2,n), stat=ierr)`

    will result in the tuple:

    ([("a",[(None,"N",None)]),("b",[("-1","m",2),(None,"n",None)])], 'ierr')
    """
    result1 = [] 
    result2 = None
    tokens = tokenize(statement)
    if not tokens.pop(0).lower() == "allocate":
        raise error.SyntaxError("expected 'allocate'")
    if tokens.pop(0) != "(":
        raise error.SyntaxError("expected '('")
    args,_  = get_top_level_operands(tokens) # : [...,"b(-1:m,n)"]
    for arg in args:
        parts,_ = get_top_level_operands(tokenize(arg),separators=["%"])
        child_tokens = tokenize(parts[-1])
        if child_tokens[0].lower() == "stat":
            if result2 != None:
                raise error.SyntaxError("expected only a single 'stat' expression")
            child_tokens.pop(0)
            if child_tokens.pop(0) != "=":
                raise error.SyntaxError("expected '='")
            result2 = "".join(child_tokens)
        else:
            varname = child_tokens.pop(0) # : "b"
            if not varname.isidentifier():
                raise error.SyntaxError("expected identifier")
            if child_tokens.pop(0) != "(":
                raise error.SyntaxError("expected '('")
            ranges,_  = get_top_level_operands(child_tokens) # : ["-1:m", "n"]
            if len(parts) > 1:
                varname = "%".join(parts[:-1]+[varname])
            var_tuple = (varname, [])
            for r in ranges:
                range_tokens,_  = get_top_level_operands(tokenize(r),
                                                      separators=[":"])
                if len(range_tokens) == 1:
                    range_tuple = (None,range_tokens[0],None)
                elif len(range_tokens) == 2:
                    range_tuple = (range_tokens[0],range_tokens[1],None)
                elif len(range_tokens) == 3:
                    range_tuple = (range_tokens[0],range_tokens[1],range_tokens[2])
                else:
                    raise error.SyntaxError("expected 1,2, or 3 colon-separated range components") 
                var_tuple[1].append(range_tuple)
            result1.append(var_tuple)
    return result1, result2

#def parse_arithmetic_expression(statement,logical_ops=False,max_recursions=0):
#    separators = []
#    R_ARITH_OPERATOR = ["**"]
#    L_ARITH_OPERATOR = "+ - * /".split(" ")
#    COMP_OPERATOR_LOWER = [
#"<= >= == /= < > .eq. .ne. .lt. .gt. .le. .ge. .and. .or. .xor. .and. .or. .not. .eqv. .neqv."
#]
#    separators +=R_ARITH_OPERATOR + L_ARITH_OPERATOR
#    if logical_ops:
#        separators += COMP_OPERATOR_LOWER
#    depth = 0
#    def inner_(statement):
#        tokens = tokenize(statement)
#        open_bracket = tokens[0] == "(":
#        top_level_operands,_ = get_top_level_operands(tokens,separators=separators,keep_separators=True)
#        if len(top_level_operands) == 1 and open_bracket:
#             if tokens[-1] == ")":
#                 raise error.SyntaxError("missing ')'")
#             result = []
#             # TODO could be complex value
#             result.append(inner_(tokens[1:-1]))
#             return
#        elif len(top_level_operands) == 1:
#            return top_level_operands
#        elif depth < max_recursions:
#            inner_(

def parse_assignment(statement,parse_rhs=False):
    tokens = tokenize(statement)
    parts,_ = get_top_level_operands(tokens,separators=["="])
    if len(parts) != 2:
        raise error.SyntaxError("expected left-hand side and right-hand side separated by '='")
    return (parts[0], parts[1])

def parse_statement_function(statement):
    lhs_expr, rhs_expr = parse_assignment(statement)
    lhs_tokens = tokenize(lhs_expr)
    name = lhs_tokens.pop(0)
    if not name.isidentifier():
        raise error.SyntaxError("expected identifier")
    if not lhs_tokens.pop(0) == "(":
        raise error.SyntaxError("expected '('")
    args, consumed_tokens = get_top_level_operands(lhs_tokens)
    for arg in args:
        if not arg.isidentifier():
            raise error.SyntaxError("expected identifier")
    for i in range(0,consumed_tokens): lhs_tokens.pop(0)
    if not lhs_tokens.pop(0) == ")":
        raise error.SyntaxError("expected ')'")
    return (name, args, rhs_expr)

# rules
def is_declaration(tokens):
    """No 'function' must be in the tokens.
    """
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

def is_derived_type_member_access(tokens):
    return len(get_top_level_operands(tokens,separators=["%"])[0]) > 1

def is_assignment(tokens):
    #assert not is_declaration_(tokens)
    #assert not is_do_(tokens)
    return len(get_top_level_operands(tokens,separators=["="])[0]) == 2

def is_pointer_assignment(tokens):
    #assert not is_ignored_statement_(tokens)
    #assert not is_declaration_(tokens)
    return len(get_top_level_operands(tokens,separators=["=>"])[0]) == 2


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
