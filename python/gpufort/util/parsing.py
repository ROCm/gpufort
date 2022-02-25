# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

COMMENT_CHARS = "!cCdD*"

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
            if len(content) > 1 and content[1] == "$":
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


def tokenize(statement, padded_size=0):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    """
    TOKENS_REMOVE = r"\s+|\t+"
    TOKENS_KEEP = r"(end|else|!\$?|[c\*]\$|[(),]|::?|=>?|<<<|>>>|[<>]=?|[/=]=|\+|-|\*|/|\.\w+\.)"
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


def next_tokens_till_open_bracket_is_closed(tokens, open_brackets=0):
    # ex:
    # input:  [  "kind","=","2","*","(","5","+","1",")",")",",","pointer",",","allocatable" ], open_brackets=1
    # result: [  "kind","=","2","*","(","5","+","1",")",")" ]
    result = []
    idx = 0
    criterion = True
    while criterion:
        tk = tokens[idx]
        result.append(tk)
        if tk == "(":
            open_brackets += 1
        elif tk == ")":
            open_brackets -= 1
        idx += 1
        criterion = idx < len(tokens) and open_brackets > 0
    # TODO throw error if open_brackets still > 0
    return result


def get_highest_level_arguments(tokens,
                                open_brackets=0,
                                separators=[","],
                                terminators=["::", "\n", "!"]):
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
        if tk == "(":
            open_brackets += 1
        elif tk == ")":
            open_brackets -= 1
    if len(current_substr):
        result.append(current_substr)
    return result

def extract_function_calls(text, func_name):
    """Extract all calls of the function `func_name` from the input text.
    :param str text: the input text.
    :param str func_name: Name of the function.
    :note: Input text must not contain any line break/continuation characters.
    :return: List of tuples that each contain the function call substring at position 0
             and the list of function call argument substrings at position 1.
    """
    result = []
    for m in re.finditer(r"{}\s*\(".format(func_name), text):
        rest_substr = next_tokens_till_open_bracket_is_closed(
            text[m.end():], 1)
        args = get_highest_level_arguments(rest_substr[:-1], 0, [","],
                                           []) # clip the ')' at the end
        end = m.end() + len(rest_substr)
        result.append((text[m.start():end], args))
    return result

def parse_use_statement(statement):
    """Extracts module name and 'only' variables
    from the statement.
    :return: A tuple of containing module name and a dict of 'only' 
    mappings (in that order).
    :raise SyntaxError: If the syntax of the expression is not as expected.

    :Example:
    
    `import mod, only: var1, var2 => var3`

    will result in the tuple:

    ("mod",{"var1":"var1","var2":"var3"])
    """
    tokens = tokenize(statement)
    only = {}
    if len(tokens) > 5 and tokens[2:5] == [",","only",":"]:
        is_original_name = True
        last = tokens[5]
        for tk in tokens[5:]:
            if tk == ",":
                is_original_name = True
            elif tk == "=>":
                is_original_name = False
            elif tk.isidentifier() and is_original_name:
                only[tk] = tk           #
                last = tk
            elif tk.isidentifier() and not is_original_name:
                only[last] = tk
            else:
                raise SyntaxError("could not parse 'use' statement")
    elif len(tokens) != 2:
        raise SyntaxError("could not parse 'use' statement")
    return tokens[1], only

def parse_declaration(fortran_statement):
    """Decomposes a Fortran declaration into its individual
    parts.

    :return: A tuple (datatype, kind, qualifiers, var-list)
             where var-list consists of triples (varname, bounds or [], right-hand-side or None)
    :raise SyntaxError: If the syntax of the expression is not as expected.
    """
    orig_tokens = tokenize(fortran_statement.lower(),
                           padded_size=10)
    tokens = orig_tokens

    idx_last_consumed_token = None
    # handle datatype
    datatype = tokens[0]
    kind = None
    DOUBLE_COLON = "::"
    try:
        if tokens[0] == "type":
            # ex: type ( dim3 )
            kind = tokens[2]
            idx_last_consumed_token = 3
        elif tokens[0:1 + 1] == ["double", "precision"]:
            datatype = " ".join(tokens[0:1 + 1])
            idx_last_consumed_token = 1
        elif tokens[1] == "*":
            # ex: integer * 4
            # ex: integer * ( 4*2 )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[2:], open_brackets=0)
            kind = "".join(kind_tokens)
            idx_last_consumed_token = 2 + len(kind_tokens) - 1
        elif tokens[1:4] == ["(","kind","("]:
            # ex: integer ( kind ( 4 ) )
            kind_tokens = next_tokens_till_open_bracket_is_closed(
                tokens[4:], open_brackets=1)
            kind = "".join(tokens[2:4]+kind_tokens)
            idx_last_consumed_token = 4 + len(kind_tokens)
        elif (tokens[1:4] == ["(","kind","="] 
             or tokens[1:4] == ["(","len","="]):
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
            raise SyntaxError("could not parse datatype")
    except IndexError:
        raise SyntaxError("could not parse datatype")
    # handle qualifiers
    try:
        tokens = tokens[idx_last_consumed_token + 1:] # remove type part tokens
        idx_last_consumed_token = None
        if tokens[0] == "," and DOUBLE_COLON in tokens:
            qualifiers = get_highest_level_arguments(tokens)
            idx_last_consumed_token = tokens.index(DOUBLE_COLON)
        elif tokens[0] == DOUBLE_COLON:
            idx_last_consumed_token = 0
        elif tokens[0] == ",":
            raise SyntaxError("could not parse qualifier list")
        qualifiers_raw = get_highest_level_arguments(tokens)

        # handle variables list
        if idx_last_consumed_token != None:
            tokens = tokens[idx_last_consumed_token+1:] # remove qualifier list tokens
        variables_raw = get_highest_level_arguments(tokens)
    except IndexError:
        raise SyntaxError("could not parse qualifier list")

    # construct declaration tree node
    qualifiers = []
    dimension_bounds = []
    for qualifier in qualifiers_raw:
        qualifier_tokens = tokenize(qualifier)
        if qualifier_tokens[0] == "dimension":
            # ex: dimension ( 1, 2, 1:n )
            qualifier_tokens = tokenize(qualifier)
            print(qualifier_tokens)
            if (len(qualifier_tokens) < 4
               or qualifier_tokens[0:2] != ["dimension","("]
               or qualifier_tokens[-1] != ")"):
                raise SyntaxError("could not parse 'dimension' qualifier")
            dimension_bounds = get_highest_level_arguments(qualifier_tokens[2:-1])
        elif qualifier_tokens[0] == "intent":
            # ex: intent ( inout )
            qualifier_tokens = tokenize(qualifier)
            if (len(qualifier_tokens) != 4 
               or qualifier_tokens[0:2] != ["intent","("]
               or qualifier_tokens[-1] != ")"
               or qualifier_tokens[2] not in ["out","inout","in"]):
                raise SyntaxError("could not parse 'intent' qualifier")
            qualifiers.append("".join(qualifier_tokens))
        elif len(qualifier_tokens)==1 and qualifier_tokens[0].isidentifier():
            qualifiers.append(qualifier)
        else:
            raise SyntaxError("could not parse qualifier '{}'".format(qualifier))
    variables = []
    # 
    for var in variables_raw:
        var_tokens = tokenize(var)
        var_name   = var_tokens.pop(0)
        var_bounds = [] 
        var_rhs    = ""
        # handle bounds
        if (len(var_tokens) > 2
           and var_tokens[0] == "("):
            if len(dimension_bounds):
                raise SyntaxError("'dimension' and variable cannot be specified both")
            bounds_tokens = next_tokens_till_open_bracket_is_closed(var_tokens)
            var_bounds = get_highest_level_arguments(bounds_tokens[1:-1])
            for i in range(0,len(bounds_tokens)):
                var_tokens.pop(0)
        if (len(var_tokens) > 1 and
           var_tokens[0] in ["=>","="]):
            var_tokens.pop(0)
            var_rhs = "".join(var_tokens)
        elif len(var_tokens):
            raise SyntaxError("could not parse '{}'".format(var_tokens))
        variables.append((var_name,var_bounds+dimension_bounds,var_rhs))
    return (datatype, kind, qualifiers, variables) 

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


def is_comment(tokens, statement):
    cond1 = tokens[0] == "!"
    cond2 = len(statement) and statement[0] in ["c", "*"]
    return cond1 or cond2


def is_cpp_directive(statement):
    return statement[0] == "#"


def is_fortran_directive(tokens, statement):
    return tokens[0] in ["!$", "*$", "c$"]


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
