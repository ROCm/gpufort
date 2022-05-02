# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# TODO use deque

import re

from . import error

def compare_ignore_case(tokens1,tokens2):
    if isinstance(tokens1,str):
        assert isinstance(tokens2,str)
        return tokens1.lower() == tokens2.lower()
    else:
        if len(tokens1) != len(tokens2):
            return False
        else:
            for i,tk in enumerate(tokens1):
                #if (tk.isidentifier() and tokens2[i]=="<ident>"
                #   or tk=="<ident>" and tokens2[i].isidentifier): 
                #    pass
                #elif tokens2[i]=="<int>":
                #    try:
                #        int(tk)
                #    except:
                #        return False 
                #elif tk=="<int>":
                #    try:
                #        int(tokens2[i])
                #    except:
                #        return False 
                #elif tk.lower() != tokens2[i].lower():
                #    return False
                if tk.lower() != tokens2[i].lower():
                    return False
            return True

def all_tokens_are_blank(tokens):
    for tk in tokens:
        if len(tk.strip("\n\r\t ")):
            return False
    return True

def check_if_all_tokens_are_blank(tokens,enable=True):
    if enable and not all_tokens_are_blank(tokens):
        raise error.SyntaxError("unexpected tokens at end of statement: {}".format(
            ",".join(["'{}'".format(tk) for tk in tokens if len(tk.strip())] )))

def pad_to_size(tokens,padded_size):
    if padded_size > 0 and len(tokens) < padded_size:
        return tokens + [""] * (padded_size - len(tokens))
    else:
        return tokens

def tokenize(statement, padded_size=0, modern_fortran=True,keepws=False):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    :param bool keepws: keep whitespaces in the resulting list of tokens.
    """
    if isinstance(statement,str):
        TOKENS_KEEP = [
            r"[\"](?:\\.|[^\"\\])*[\"]",
            r"[\'](?:\\.|[^\'\\])*[\']",
            r"\bend(?:if|do)?\b",
            r"\belse(?:if)?\b",
            r"[();,%]",
            r"::?",
            r"<<<",
            r">>>",
            r"[<>]=?",
            r"[\/=]=",
            r"=>?",
            r"\+",
            r"-",
            r"\*",
            r"\/",
            r"\&",
            r"\.\w+\.",
            r"[\s\t\r\n]+",
        ]
        if modern_fortran:
            TOKENS_KEEP.append(r"![@\$]?")
        else:
            TOKENS_KEEP.append(r"^[c\*][@\$]?")
        # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
        # is not captured.
        keep_pattern = "".join(["(","|".join(TOKENS_KEEP),")"])
        
        tokens = re.split(keep_pattern, statement, 0, re.IGNORECASE)
        result = []
        for tk in tokens:
            if tk.lower() in ["endif", "elseif", "enddo"]:
                result.append(tk[:-2])
                result.append(tk[-2:])
            elif len(tk) and (keepws or len(tk.strip())):
                result.append(tk)
        return pad_to_size(result,padded_size)
    elif isinstance(statement,list):
        if keepws:
            return pad_to_size(statement,padded_size)
        else:
            return pad_to_size([tk for tk in statement if len(tk.strip())],padded_size)
    else:
        raise Exception("input must be either a str or a list of strings")

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

def split_fortran_line(line,modern_fortran=True):
    """Decomposes a Fortran line into preceding whitespace,
     statement part, comment part, and trailing whitespace (including newline char).
     """
    tokens = tokenize(line,modern_fortran=modern_fortran,keepws=True)
    if len(tokens) and not len(tokens[0].strip("  \r\t\n")):
        preceding_ws = tokens.pop(0)
    else: 
        preceding_ws = ""
    if len(tokens) and not len(tokens[-1].strip("  \r\t\n")):
        trailing_ws = tokens.pop(-1)
    else: 
        trailing_ws = ""
    statement_tokens = []
    comment_tokens = []
    if len(tokens):
        if (not modern_fortran and tokens[0].lower() in ["c","*"]
             or modern_fortran and tokens[0] == "!"):
            comment_tokens = tokens
        elif modern_fortran:
            while len(tokens):
                if tokens[0] == "!": 
                    break
                else:
                    statement_tokens.append(tokens.pop(0))
            comment_tokens = tokens # what is left must be an inline comment
        else:
            statement_tokens = tokens
    return preceding_ws, "".join(statement_tokens), "".join(comment_tokens), trailing_ws

def detect_line_starts(lines,modern_fortran=True):
    """Fortran statements can be broken into multiple lines 
    via the '&' characters. This routine detects in which line a statement
    (or multiple statements per line) begins.
    The difference between the line numbers of consecutive entries
    is the number of lines the first statement occupies.
    """
    p_directive_continuation = re.compile(r"[!c\*][@\$]\w+\&")

    # 1. save multi-line statements (&) in buffer
    buffering = False
    line_starts = []
    for lineno, line in enumerate(lines, start=0):
        # Continue buffering if multiline CUF/ACC/OMP statement
        _,stmt_or_dir,comment,_ = split_fortran_line(line)
        stmt_or_dir_tokens = tokenize(stmt_or_dir)
        # continuation at line begin
        if (len(stmt_or_dir_tokens) > 3
           and (modern_fortran and stmt_or_dir_tokens[0] in ["!$","!@"]
               or not modern_fortran and stmt_or_dir_tokens[0].lower() in ["c$","c@","*$","*@"])):
            buffering = buffering or stmt_or_dir_tokens[2] == "&" 
        if not buffering:
            line_starts.append(lineno)
        if len(stmt_or_dir) and stmt_or_dir_tokens[-1] in ['&', '\\']:
            buffering = True
        elif len(stmt_or_dir):
            buffering = False
    line_starts.append(len(lines))
    return line_starts


def relocate_inline_comments(lines,modern_fortran=True):
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
          split_fortran_line(line,modern_fortran)
        stmt_or_dir_tokens = tokenize(stmt_or_dir)
        if len(stmt_or_dir_tokens) and stmt_or_dir_tokens[-1] == "&":
            in_multiline_statement = True
        if in_multiline_statement and len(comment):
            if len(stmt_or_dir):
                result.append("".join([indent, stmt_or_dir, trailing_ws]))
            comment_buffer.append("".join([indent, comment, trailing_ws]))
        elif len(line.strip()): # only append non-empty lines
            result.append(line)
        if len(stmt_or_dir_tokens) and stmt_or_dir_tokens[-1] != "&":
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

def _join_tokens(tokens,no_ws=True):
    """Join tokens, insert whitespace between identifiers if
    the tokens do not contain whitespace.
    :param bool no_ws: The tokens contain whitespace, defaults to True.
    """
    if no_ws:
        result = ""
        last = ""
        for tk in tokens:
            if tk.isidentifier() and last.isidentifier():
                result += " "
            result += tk
            last = tk
        return result
    else:
        return "".join(tokens)

def get_top_level_operands(tokens,
                           separators=[","],
                           terminators=["::", "\n", "!",""],
                           brackets=("(",")"),
                           keep_separators=False,
                           join_operand_tokens=True,
                           no_ws=True):
    """:return: Tuple of highest level arguments and number of consumed tokens
             in that order.
       :param bool join_operand_tokens: If the tokens of the individual operands should
              be joined (or be returned as list). Defaults to True.
        :param bool no_ws: The tokens contain whitespace, defaults to True.
       :note: terminators are excluded from the number of consumed tokens.
    """
    # ex:
    # input: ["parameter",",","intent","(","inout",")",",","dimension","(",":",",",":",")","::"]
    # result : ["parameter", "intent(inout)", "dimension(:,:)" ], i.e. terminators are excluded
    result = []
    idx = 0
    current_operand = []
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
            if len(current_operand):
                if join_operand_tokens:
                    result.append(_join_tokens(current_operand,no_ws))
                else:
                    result.append(current_operand)
            if keep_separators:
                result.append(tk)
            current_operand = []
        elif tk_lower in terminators or open_brackets < 0:
            criterion = False
            num_consumed_tokens -= 1
        else: 
            current_operand.append(tk)
    if len(current_operand):
        if join_operand_tokens:
            result.append(_join_tokens(current_operand,no_ws))
        else:
            result.append(current_operand)
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
                                           [], no_ws=False)
        end = m.end() + len(rest_substr)
        result.append((text[m.start():end], args))
    return result

def parse_function_statement(statement):
    """Fortran function and subroutine statements can take
    various forms, e.g.:
    ```
    function f1(i) result (j)
    integer function f2(i) result (j)
    integer function f3(i)
    function f4(i)
    ```
    where, in case of function statements, a return value type and name might be specified
    as part of the signature or not.

    Additionally, there might be other attributes in the prefix part in front of `function` keyword, 
    such as `pure` or `recursive`. Return value type and all other attributes might 
    only occur once in the prefix part.
    In addition to the `result(<varname>)` modifier there could also
    be a `bind(c,"<cname>")` qualifier.
    """
    name, kind, dummy_args, modifiers, attributes = None, None, [], [], []
    result_type, result_type_kind, result_name = None, None, None
    bind_c, bind_c_name = False, None
    #
    type_begin = [
      "character", "integer", "logical", "real",
      "complex", "double", "type"
    ]
    anchors = ["subroutine","function"]
    simple_modifiers = ["recursive","pure","elemental"]

    tokens = tokenize(statement,padded_size=10)
    criterion = len(tokens)
    # prefix
    while criterion:
        if tokens[0].lower() in type_begin:
            result_type, f_len, result_type_kind, params, num_consumed_tokens =\
                    _parse_datatype(tokens)
            tokens=tokens[num_consumed_tokens:]
        else:
            tk = tokens.pop(0)
            criterion = len(tokens)
            if tk.lower() in simple_modifiers:
                modifiers.append(tk)
            elif tk.lower() == "attributes":
                if not tokens.pop(0) == "(":
                    raise error.SyntaxError("expected '(' after 'attributes'")
                attributes, num_consumed_tokens =  get_top_level_operands(tokens)
                if not len(attributes):
                    raise error.SyntaxError("expected at least one attribute in the 'attributes' list")
                tokens = tokens[num_consumed_tokens:]
                if not tokens.pop(0) == ")":
                    raise error.SyntaxError("expected ')' after 'attributes' list")
                criterion = len(tokens)
            elif tk.lower() in anchors:
                kind = tk
                break
            elif len(tk.strip()):
                raise error.SyntaxError("unexpected prefix token: {}".format(tk))
    # name and dummy args
    if kind == None:
        raise error.SyntaxError("expected 'function' or 'subroutine'")
    name = tokens.pop(0)
    if kind.lower() == "function":
        result_name = name
    if not name.isidentifier():
        raise error.SyntaxError("expected identifier after '{}'".format(kind))
    if len(tokens):
        if tokens.pop(0) == "(":
            dummy_args, num_consumed_tokens = get_top_level_operands(tokens)  
            tokens = tokens[num_consumed_tokens:]
            if not tokens.pop(0) == ")":
                raise error.SyntaxError("expected ')' after dummy argument list")
    # result and bind
    criterion = len(tokens)
    while criterion:
        tk = tokens.pop(0)
        criterion = len(tokens)
        suffix_kind = tk.lower()
        if suffix_kind in ["bind","result"]:
            if not tokens.pop(0) == "(":
                raise error.SyntaxError("expected '(' after '{}'".format(suffix_kind))
            operands, num_consumed_tokens =  get_top_level_operands(tokens)
            if suffix_kind == "bind":
                if len(operands) in [1,2]:
                    bind_c = True
                else:
                    raise error.SyntaxError("'bind' takes one or two arguments")
                if not operands[0].lower() == "c":
                    raise error.SyntaxError("expected 'c' as first 'bind' argument")
                if len(operands) == 2:
                    bind_c_name_parts = operands[1].split("=")
                    if len(bind_c_name_parts) != 2:
                        raise error.SyntaxError("expected 'name=\"<label>\"' (or single quotes) as 2nd argument of 'bind'")
                    else:
                        bind_c_name = bind_c_name_parts[1].strip("\"'")
                        if (bind_c_name_parts[0].lower() != "name"
                                or not bind_c_name.isidentifier()):
                            raise error.SyntaxError("expected 'name=\"<label>\"' as 2nd argument of 'bind'")
            elif suffix_kind == "result":
                if not len(operands) == 1:
                    raise error.SyntaxError("'result' takes one argument")
                if not operands[0].isidentifier():
                    raise error.SyntaxError("'result' argument must be identifier")
                result_name = operands[0]
            tokens = tokens[num_consumed_tokens:]
            if not tokens.pop(0) == ")":
                raise error.SyntaxError("expected ')' after 'attributes' list")
        elif len(tk.strip()):
            raise error.SyntaxError("unexpected suffix token: {}".format(tk))
        criterion = len(tokens)
    check_if_all_tokens_are_blank(tokens)
    return (
        kind, name, dummy_args, modifiers, attributes,
        (result_type, result_type_kind, result_name),
        (bind_c, bind_c_name)
    )



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

#def _parse_datatype_v1(tokens):
#    # TODO treat len of CHARACTER explicitly
#    # handle datatype
#    tokens = pad_to_size(tokens,6)
#    datatype = tokens[0]
#    idx_last_consumed_token = 0
#    kind = None
#    DOUBLE_COLON = "::"
#    
#    if compare_ignore_case(tokens[0],"type"):
#        # ex: type ( dim3 )
#        kind = tokens[2]
#        idx_last_consumed_token = 3
#    elif compare_ignore_case(tokens[0:2],["double", "precision"]):
#        datatype = " ".join(tokens[0:2])
#        idx_last_consumed_token = 1
#    elif tokens[1] == "*":
#        # ex: integer * 4
#        # ex: integer * ( 4*2 )
#        if tokens[2] == "(":
#            kind_tokens = next_tokens_till_open_bracket_is_closed(
#                tokens[3:], open_brackets=1)[:-1]
#            kind = "".join(kind_tokens)
#            idx_last_consumed_token = 2 + len(kind_tokens) + 1 # )
#        else:
#            kind = tokens[2] 
#            idx_last_consumed_token = 2
#    elif compare_ignore_case(tokens[1:4],["(","kind","("]):
#        # ex: integer ( kind ( 4 ) )
#        kind_tokens = next_tokens_till_open_bracket_is_closed(
#            tokens[4:], open_brackets=2)
#        kind = "".join(tokens[2:4]+kind_tokens[:-1])
#        idx_last_consumed_token = 4 + len(kind_tokens) - 1
#    elif (compare_ignore_case(tokens[1:4], ["(","kind","="])
#         or compare_ignore_case(tokens[1:4],["(","len","="])):
#        # ex: integer ( kind = 4 )
#        kind_tokens = next_tokens_till_open_bracket_is_closed(
#            tokens[4:], open_brackets=1)
#        kind = "".join(kind_tokens[:-1])
#        idx_last_consumed_token = 4 + len(kind_tokens) - 1
#    elif tokens[1] == "(":
#        # ex: integer ( 4 )
#        # ex: integer ( 4*2 )
#        kind_tokens = next_tokens_till_open_bracket_is_closed(
#            tokens[2:], open_brackets=1)
#        kind = "".join(kind_tokens[:-1])
#        idx_last_consumed_token = 2 + len(kind_tokens) - 1
#    elif tokens[1] in [",", DOUBLE_COLON] or tokens[1].isidentifier():
#        # ex: integer ,
#        # ex: integer ::
#        # ex: integer a
#        idx_last_consumed_token = 0
#    elif not all_tokens_are_blank(tokens[1:]):
#        raise error.SyntaxError("could not parse datatype")
#    return datatype, kind, idx_last_consumed_token+1

def _parse_raw_qualifiers_and_rhs(tokens):
    """:return: Triple of qualifiers and variables as list of raw strings plus
                the number of consumed tokens (in that order).
    """
    DOUBLE_COLON = "::"
    try:
        qualifiers_raw = []
        idx_last_consumed_token = None
        if tokens[0] == "," and DOUBLE_COLON in tokens: # qualifier list
            qualifiers_raw,_  = get_top_level_operands(tokens)
            idx_last_consumed_token = tokens.index(DOUBLE_COLON)
        elif tokens[0] == DOUBLE_COLON: # variables begin afterwards
            idx_last_consumed_token = 0
        elif tokens[0].isidentifier(): # variables begin directly
            idx_last_consumed_token = -1
        else:
            raise error.SyntaxError("could not parse qualifiers and right-hand side")
        # handle variables list
        num_consumed_tokens = idx_last_consumed_token+1
        tokens = tokens[num_consumed_tokens:]
        variables_raw, num_consumed_tokens2  = get_top_level_operands(tokens)
        num_consumed_tokens += num_consumed_tokens2
        if not len(variables_raw):
            raise error.SyntaxError("expected at least one entry in right-hand side")
        return qualifiers_raw, variables_raw, num_consumed_tokens
    except IndexError:
        raise error.SyntaxError("could not parse qualifiers and right-hand side")

def parse_type_statement(statement):
    """Parse a Fortran type statement.
    :return: List 
    """
    tokens = tokenize(statement,10)
    if tokens.pop(0).lower() != "type":
        raise error.SyntaxError("expected 'type'")
    attributes, rhs, num_consumed_tokens = _parse_raw_qualifiers_and_rhs(tokens)
    tokens = tokens[num_consumed_tokens:] # remove qualifier list tokens
    check_if_all_tokens_are_blank(tokens)
    if len(rhs) != 1:
        raise error.SyntaxError("expected 1 identifier with optional parameter list")
    rhs_tokens = tokenize(rhs[0])
    name = rhs_tokens.pop(0)
    if not name.isidentifier():
        raise error.SyntaxError("expected identifier")
    params = []
    if len(rhs_tokens) and rhs_tokens.pop(0)=="(":
        params,num_consumed_tokens2 = get_top_level_operands(rhs_tokens)  
        rhs_tokens = rhs_tokens[num_consumed_tokens2:]
        if not rhs_tokens.pop(0) == ")":
            raise error.SyntaxError("expected ')'")
        if not len(params):
            raise error.SyntaxError("expected at least 1 parameter")
    return name, attributes, params

def _parse_fortran_argument(statement):
    tokens = tokenize(statement)
    parts,num_consumed_tokens = get_top_level_operands(tokens,separators="=")
    if len(parts) == 2:
        return (parts[0],parts[1])
    elif len(parts) == 1:
        return (None,parts[0])
    else:
        raise error.SyntaxError("expected single expression or named argument expression")

def _map_fortran_args_to_positional_args(fortran_args,
                                         positional_args=None,
                                         defaults=None,
                                         allow_named_args=True):
    """:return: List that contains value expressions per detected positional argument or None.
                If `positional_args` is set to None or not specified, this function
                returns None too. In this case, the function checks if all named arguments
                are specified after the last positional argument; it raises an `error.SyntaxError`
                if not so.
    :param fortran_args: A Fortran arg list either as string, list of str or 
                         list of fortran arg tuples.
    :param list positional_args: List of positional arg names or None.
    :param list defaults: List of default values to set or None. Only used if
                          positional arg names are not None.
 
    :throws: error.SyntaxError, ValueError
    """
    args1 = None
    if isinstance(fortran_args, str):
        args1,_ = get_top_level_operands(tokenize(fortran_args))
    elif isinstance(fortran_args, list):
        if len(fortran_args) and isinstance(fortran_args[0],str):
            args1 = fortran_args
        elif len(fortran_args) and isinstance(fortran_args[0],tuple):
            fortran_arg_list = fortran_args 
        else:
            raise ValueError("type of 'fortran_args' must be either str, list of str, or list of 2-tuples with str or None elements")
    else:
        raise ValueError("type of 'fortran_args' must be either str, list of str, or list of 2-tuples with str or None elements")
    if args1 != None:
        fortran_arg_list = [_parse_fortran_argument(expr) for expr in args1]
    #
    values = None
    if positional_args != None:
        if len(fortran_arg_list) > len(positional_args):
            raise error.SyntaxError("expected no more than {} arguments, but {} were passed".format(len(positional_args),len(fortran_arg_list)))
        if defaults == None:
            values = [None]*len(positional_args)
        elif len(defaults) != len(positional_args):
            raise ValueError("len(defaults) != len(positional_args)")
        else:
            values = defaults
            
    consumed_positional_args = []
    consumed_named_args = []
    all_unnamed = True 
    for i,pair in enumerate(fortran_arg_list):
        name,value = pair
        if name == None:
            if not all_unnamed:
                raise error.SyntaxError("positional argument after named argument(s)")
            if positional_args != None:
               name = positional_args[i]
               consumed_positional_args.append(name) 
               values[i] = value 
        elif not allow_named_args:
            raise error.SyntaxError("no named arguments allowed")
        else:
            all_unnamed = False
            name_lower = name.lower()
            if name_lower in consumed_positional_args:
                raise error.SyntaxError("argument '{}' already appeared as positional argument".format(name_lower))
            elif name_lower in consumed_named_args:
                raise error.SyntaxError("argument '{}' already appeared as named argument".format(name_lower))
            if positional_args != None:
                if name_lower not in positional_args:
                    raise error.SyntaxError("invalid argument name '{}'".format(name))
                idx = positional_args.index(name_lower)
                values[idx] = value
            consumed_named_args.append(name_lower)
    return values

def _parse_f77_character_type(statement,parse_all=False):
    tokens = tokenize(statement,padded_size=3)
    char_type = tokens[0]
    char_kind = None 
    char_len  = "1"
    if compare_ignore_case(tokens[0:3],["character","*","("]):
        total_num_consumed_tokens = 3
        args, num_consumed_tokens = get_top_level_operands(tokens[3:])
        total_num_consumed_tokens += num_consumed_tokens
        if tokens[total_num_consumed_tokens:total_num_consumed_tokens+1] != [")"]:
            raise error.SyntaxError("missing ')'")
        total_num_consumed_tokens += 1
        check_if_all_tokens_are_blank(tokens[total_num_consumed_tokens:],parse_all)
        if len(args) == 1:
            char_len = _map_fortran_args_to_positional_args(args,
                                                            positional_args=["len"],
                                                            defaults=["1"],allow_named_args=False)[0]
            return (char_type, char_len, None, [], total_num_consumed_tokens)
        else:
            raise error.SyntaxError("expected a single unnamed 'len' expression")        
    elif compare_ignore_case(tokens[0:2],["character","*"]):
        char_len = tokens[2]
        if not char_len.isidentifier():
            try:
               int(char_len) 
            except ValueError:
                raise error.SyntaxError("if no parantheses are used, character len argument must be identifier or integer")
        check_if_all_tokens_are_blank(tokens[3:],parse_all)
        return (char_type, char_len, None, [], 3)
    elif compare_ignore_case(tokens[0],"character"):
        check_if_all_tokens_are_blank(tokens[1:],parse_all)
        return (char_type, char_len, None, [], 1)
    else:
        raise error.SyntaxError("expected 'character'") 
            
def _parse_character_type(statement,parse_all=False):
    tokens = tokenize(statement,padded_size=2)
    if compare_ignore_case(tokens[0:2],["character","("]):
        total_num_consumed_tokens = 2 
        args, num_consumed_tokens = get_top_level_operands(tokens[total_num_consumed_tokens:])
        total_num_consumed_tokens += num_consumed_tokens
        if tokens[total_num_consumed_tokens:total_num_consumed_tokens+1] != [")"]:
            raise error.SyntaxError("missing ')'")
        total_num_consumed_tokens += 1
        check_if_all_tokens_are_blank(tokens[total_num_consumed_tokens:],parse_all)
        if len(args) in [1,2]:
            values = _map_fortran_args_to_positional_args(args,["len","kind"],defaults=["1",None])
            char_len, char_kind = values
            return (tokens[0], char_len, char_kind, [], total_num_consumed_tokens)
        else:
            raise error.SyntaxError("expected 1 or 2 comma-separated character length and kind expressions")
    else:
        return _parse_f77_character_type(tokens,parse_all)

__basic_types = ["logical","integer","real","complex"]

def _parse_f77_basic_type(statement,parse_all=False):
    tokens = tokenize(statement,padded_size=3)
    if compare_ignore_case(tokens[0:2],["double","precision"]):
        check_if_all_tokens_are_blank(tokens[2:],parse_all)
        return ("real", None, "c_double", [], 2)
    elif compare_ignore_case(tokens[0:2],["double","complex"]):
        check_if_all_tokens_are_blank(tokens[2:],parse_all)
        return ("complex", None, "c_double_complex", [], 2)
    elif tokens[0].lower() in __basic_types:
        basic_type = tokens[0]
        if compare_ignore_case(tokens[1:3],["*","("]):
            total_num_consumed_tokens = 3
            args, num_consumed_tokens = get_top_level_operands(tokens[3:])
            total_num_consumed_tokens += num_consumed_tokens
            if tokens[total_num_consumed_tokens:total_num_consumed_tokens+1] != [")"]:
                raise error.SyntaxError("missing ')'")
            total_num_consumed_tokens += 1
            check_if_all_tokens_are_blank(tokens[total_num_consumed_tokens:],parse_all)
            if len(args) == 1:
                basic_kind = _map_fortran_args_to_positional_args(args,
                                                                  positional_args=["kind"],
                                                                  defaults=[None],allow_named_args=False)[0]
                return (basic_type, None, basic_kind, [], total_num_consumed_tokens)
            else:
                raise error.SyntaxError("expected a single unnamed 'kind' expression")        
        elif compare_ignore_case(tokens[1:2],["*"]):
            basic_kind = tokens[2]
            if not basic_kind.isidentifier():
                try:
                   int(basic_kind) 
                except ValueError:
                    raise error.SyntaxError("if no parantheses are used, kind argument must be identifier or integer")
            check_if_all_tokens_are_blank(tokens[3:],parse_all)
            return (basic_type, None, basic_kind, [], 3)
        else:
            check_if_all_tokens_are_blank(tokens[1:],parse_all)
            return (basic_type, None, None, [], 1)
    else:
            raise error.SyntaxError("expected 'character'") 

def _parse_basic_type(statement,parse_all=False):
    """:note: character is handleded in a dedicated function."""
    tokens = tokenize(statement,padded_size=2)
    if (tokens[0].lower() in __basic_types
       and compare_ignore_case(tokens[1:2],["("])):
        total_num_consumed_tokens = 2 
        args, num_consumed_tokens = get_top_level_operands(tokens[total_num_consumed_tokens:])
        total_num_consumed_tokens += num_consumed_tokens
        if tokens[total_num_consumed_tokens:total_num_consumed_tokens+1] != [")"]:
            raise error.SyntaxError("missing ')'")
        total_num_consumed_tokens += 1
        check_if_all_tokens_are_blank(tokens[total_num_consumed_tokens:],parse_all)
        if len(args) == 1:
            basic_kind = _map_fortran_args_to_positional_args(args,["kind"],defaults=[None])[0]
            return (tokens[0],None,basic_kind,[],total_num_consumed_tokens)
        else:
            raise error.SyntaxError("expected a single kind expressions")
    else:
        return _parse_f77_basic_type(tokens,parse_all)

def _parse_derived_type(statement,parse_all=True):
    tokens = tokenize(statement,padded_size=2)
    if compare_ignore_case(tokens[0:2],["type","("]):
        total_num_consumed_tokens = 2
        types,num_consumed_tokens = get_top_level_operands(tokens[total_num_consumed_tokens:],join_operand_tokens=False)
        total_num_consumed_tokens += num_consumed_tokens
        if tokens[total_num_consumed_tokens:total_num_consumed_tokens+1] != [")"]:
            raise error.SyntaxError("missing ')'")
        total_num_consumed_tokens += 1
        check_if_all_tokens_are_blank(tokens[total_num_consumed_tokens:],parse_all)
        if len(types) == 1:
            first_entry = types[0]
            if len(first_entry) == 1 and first_entry[0].isidentifier():
                return (tokens[0],None,first_entry[0],[],total_num_consumed_tokens)
            elif len(first_entry) >= 2 and first_entry[0].isidentifier() and first_entry[1] == "(":
                params,num_params_tokens = get_top_level_operands(first_entry[2:])
                if first_entry[2+num_params_tokens:] != [")"]:
                    raise error.SyntaxError("missing ')'")
                check_if_all_tokens_are_blank(first_entry[2+num_params_tokens+1:],parse_all)
                return (tokens[0],None,first_entry[0],params,total_num_consumed_tokens)
            else:
                raise error.SyntaxError("expected identifier or identifier plus parameter list")
        else:
            raise error.SyntaxError("expected single identifier or identifier plus parameter list")
    else:
        raise error.SyntaxError("expected 'type ('")

def _parse_datatype(statement,parse_all=False):
    tokens = tokenize(statement,padded_size=1)
    if compare_ignore_case(tokens[0],"type"):
        return _parse_derived_type(tokens,parse_all)
    elif compare_ignore_case(tokens[0],"character"):
        return _parse_character_type(tokens,parse_all)
    else:
        return _parse_basic_type(tokens,parse_all)
        

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
    is_character_declaration = tokens[0].lower() == "character"
    is_f77_character_declaration = compare_ignore_case(tokens[0:2],["character","*"])

    # handle datatype
    datatype, length, kind, params, num_consumed_tokens = _parse_datatype(tokens)
    
    datatype_raw = "".join(tokens[:num_consumed_tokens])
   
    # parse raw qualifiers and variables
    tokens = tokens[num_consumed_tokens:] # remove type part tokens
    try:
        qualifiers_raw, variables_raw, num_consumed_tokens = _parse_raw_qualifiers_and_rhs(tokens)
        tokens = tokens[num_consumed_tokens:] # remove qualifier list tokens
    except error.SyntaxError: # superfluous comma allowed between character*<expr> and first identifier
        if is_f77_character_declaration and tokens[0] == ",":
            tokens = tokens[1:]
            qualifiers_raw, variables_raw, num_consumed_tokens = _parse_raw_qualifiers_and_rhs(tokens)
            tokens = tokens[num_consumed_tokens:] # remove qualifier list tokens
        else:
            raise
    check_if_all_tokens_are_blank(tokens)
    # analyze qualifiers
    qualifiers = []
    dimension_bounds = []
    for qualifier in qualifiers_raw:
        qualifier_tokens       = tokenize(qualifier)
        if qualifier_tokens[0].lower() == "dimension":
            # ex: dimension ( 1, 2, 1:n )
            qualifier_tokens = tokenize(qualifier)
            if (len(qualifier_tokens) < 4
               or not compare_ignore_case(qualifier_tokens[0:2],["dimension","("])
               or qualifier_tokens[-1] != ")"):
                raise error.SyntaxError("could not parse 'dimension' qualifier")
            dimension_bounds,_  = get_top_level_operands(qualifier_tokens[2:-1])
        elif qualifier_tokens[0].lower() == "intent":
            # ex: intent ( inout )
            qualifier_tokens = tokenize(qualifier)
            if (len(qualifier_tokens) != 4 
               or not compare_ignore_case(qualifier_tokens[0:2],["intent","("])
               or qualifier_tokens[-1] != ")"
               or qualifier_tokens[2].lower() not in ["out","inout","in"]):
                raise error.SyntaxError("could not parse 'intent' qualifier")
            qualifiers.append("".join(qualifier_tokens))
        elif len(qualifier_tokens)==1 and qualifier_tokens[0].isidentifier():
            qualifiers.append(qualifier)
        else:
            raise error.SyntaxError("could not parse qualifier '{}'".format(qualifier))
    # analyze variables
    variables = []
    for var in variables_raw:
        separator = "=>" if "pointer" in qualifiers else "="
        assignment_operands,_ = get_top_level_operands(tokenize(var),
                                                       separators=[separator])
        if len(assignment_operands) == 1:
            var_lhs = assignment_operands[0]
            var_rhs = None
        elif len(assignment_operands) == 2:
            var_lhs,var_rhs = assignment_operands
        else:
            raise error.SyntaxError("expected single variable expression or (pointer) assigment")
        if is_character_declaration:
            length_operands,_ = get_top_level_operands(tokenize(var_lhs),
                                                       separators=["*"])
            if len(length_operands) == 2:
                var_lhs, length = length_operands 
        var_lhs_tokens = tokenize(var_lhs)
        var_name = var_lhs_tokens[0]
        if not var_name.isidentifier():
            raise error.SyntaxError("expected identifier")
        if (len(var_lhs_tokens) > 2
           and var_lhs_tokens[1] == "("):
            var_bounds, num_consumed_tokens = get_top_level_operands(var_lhs_tokens[2:])
            if not len(var_bounds):
                raise error.SyntaxError("expected at least one argument for variable '{}'".format(var_name))
            if var_lhs_tokens[2+num_consumed_tokens:] != [")"]:
                raise error.SyntaxError("missing ')'")
        elif len(var_lhs_tokens) > 1:
            raise error.SyntaxError("unexpected tokens after '{}': '{}'".format(var_name,"','".join(var_lhs_tokens[1:])))
        else:
            var_bounds = []
        variables.append((var_name,var_bounds,var_rhs))
    return (datatype, length, kind, params, qualifiers, dimension_bounds, variables, datatype_raw, qualifiers_raw) 

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
    elif len(tk.strip()):
        raise error.SyntaxError("unexpected token: '{}'".format(tk))
    return name, attributes, parameters

def parse_public_or_private_statement(statement,kind):
    """Parses statements of the form:
    ```
    private
    public
    public identifier-list
    private :: identifier-list
    public  :: identifier-list
    public operator(<op>)
    public assignment(<assignment-op>)
    ```
    :return: tuple of visibility kind ('private','public')
             and the list of identifier strings (might be empty).
    """
    tokens      = tokenize(statement,padded_size=1)
    kind_expr   = tokens.pop(0)
    identifiers = []
    operators   = []
    assignments = []
    if kind_expr.lower() != kind:
        raise error.SyntaxError("expected '{}'".format(kind))
    if len(tokens) > 1 and tokens[0] == "::":
        tokens.pop(0)
    if len(tokens) and tokens[0].isidentifier():
        operands, num_consumed_tokens = get_top_level_operands(tokens,join_operand_tokens=False)
        for expr in operands:
            if len(expr) == 1 and expr[0].isidentifier():
                identifiers.append(expr[0])
            elif (len(expr) == 4
               and compare_ignore_case(expr[0:2],["operator","("])
               and expr[3] == ")"):
                # TODO check if operator is valid
                identifiers.append("".join(expr))
            elif (len(expr) == 4
               and compare_ignore_case(expr[0:2],["assignment","("])
               and expr[3] == ")"):
                # TODO check if assignment is valid
                identifiers.append("".join(expr))
            else:
                raise error.SyntaxError("expected identifier or 'operator' + '(' + operator expression + ')'")
        tokens = tokens[num_consumed_tokens:] 
    check_if_all_tokens_are_blank(tokens)
    return kind_expr, identifiers

def parse_public_statement(statement):
    """:see: parse_public_or_private_statement"""
    return parse_public_or_private_statement(statement,"public")

def parse_private_statement(statement):
    """:see: parse_public_or_private_statement"""
    return parse_public_or_private_statement(statement,"private")

def expand_letter_range(expr):
    """Expand expressions such as 'a-z' and trivial cases such as 'a'.
    :return: The expanded list of characters in the given range.
    """
    result = set()
    parts = expr.split("-")
    if (len(parts) == 1
       and len(parts[0])==1 and parts[0].isalpha()):
        return parts
    if (len(parts) == 2
       and len(parts[0])==1 and parts[0].isalpha()
       and len(parts[1])==1 and parts[1].isalpha()):
        ord1 = ord(parts[0])
        ord2 = ord(parts[1])
        if ord2 == ord1:
            return [parts[0]]
        elif ord1 < ord2:
            return [chr(i) for i in range(ord1,ord2+1)]
        else:
            raise error.SyntaxError("do not support parameter expressions in letter range."
                                    +"Did you mean to write '{1}-{0}' instead of '{0}-{1}'?".format(
                                        parts[0],parts[1]))
        
    else:
        raise error.SyntaxError("expected expression '<letter>' or '<letter1>-<letter2>'")
    
def parse_implicit_statement(statement):
    """Parses statements of the form:
    ```
    implicit none
    implicit type(mytype) (a,e,i,o,u)
    implicit character (c)           
    implicit integer(mykind) (j,k,m,n)       
    implicit real (b, d, f-h, p-t, v-z)
    ```
    :return: A list of implicit naming rules. Each naming
             rule is a tuple of the Fortran type, length, and kind plus the associated first letters.
             Function will return successfully with empty list for statement `implicit none`.
             If there is no implicit rule or 'none' specified after 'implicit',
             this method will throw error.SyntaxError.
    """
    tokens = tokenize(statement,padded_size=2)    
    implicit_specs = []
    if tokens.pop(0).lower() != "implicit":
        raise error.SyntaxError("expected 'implicit'")
    rules,num_consumed_tokens = get_top_level_operands(tokens)
    tokens = tokens[num_consumed_tokens:]
    check_if_all_tokens_are_blank(tokens)
    if len(rules)==1 and rules[0].lower() == "none":
        implicit_specs.append((None,None,None,[]))
        check_if_all_tokens_are_blank(rules[1:])
    elif len(rules):
        taken_letters = set()
        # example: implicit character (c), integer(mykind) (j,k,m,n)       
        for rule in rules:
            rule_tokens  = tokenize(rule)
            datatype_tokens = next_tokens_till_open_bracket_is_closed(
                    rule_tokens,open_brackets=1,brackets=(None,")"))
            if len(datatype_tokens) == len(rule_tokens):
                # example: implicit character (c)           
                datatype_tokens = rule_tokens[:1]
                first_letter_tokens = rule_tokens[1:]
            else:
                # example: implicit integer(mykind) (j,k,m,n)       
                first_letter_tokens = rule_tokens[len(datatype_tokens):]
            f_type, f_len, f_kind, params, num_consumed_tokens = _parse_datatype(datatype_tokens)
            assigned_first_letters = []
            check_if_all_tokens_are_blank(datatype_tokens[num_consumed_tokens:])
            #rule_tokens = rule_tokens[num_consumed_tokens:]
            #letters = []
            if (len(first_letter_tokens) and first_letter_tokens[0] == "("
               and first_letter_tokens[-1] == ")"):
                letter_range_expressions,num_consumed_tokens = get_top_level_operands(first_letter_tokens[1:-1])
                check_if_all_tokens_are_blank(first_letter_tokens[num_consumed_tokens+2:])
                
                for expr in letter_range_expressions:
                    letter_range = expand_letter_range(expr)
                    for letter in letter_range:
                        if letter not in taken_letters:
                            assigned_first_letters.append(letter) 
                        else:
                            raise error.SyntaxError("letter '{}' already has an implicit type".format(letter))
                        taken_letters.add(letter)
                implicit_specs.append((f_type,None,f_kind,assigned_first_letters))
            else:
                raise error.SyntaxError("expected list of implicit naming rules")
        pass
    else:
        raise error.SyntaxError("expected 'none' or at least one implicit rule specification")
    return implicit_specs

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
    """Parses an OpenACC directive, returns a 4-tuple of its sentinel, its kind (as list of tokens),
    its arguments and its clauses (as list of tokens). 

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
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'call'")
    kernel_name = tokens.pop(0)
    # parameters
    if not kernel_name.isidentifier():
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'identifier'")
    if tokens.pop(0) != "<<<":
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '<<<'")
    params_tokens = next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, brackets=("<<<",">>>"), keepend=False)
    for i in range(0,len(params_tokens)):
        tokens.pop(0)
    if tokens.pop(0) != ">>>":
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '>>>'")
    # arguments
    if tokens.pop(0) != "(":
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '('")
    args_tokens = next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, keepend=False)
    for i in range(0,len(args_tokens)):
        tokens.pop(0)
    if tokens.pop(0) != ")":
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected ')'")
    params,_  = get_top_level_operands(params_tokens,terminators=[])
    args,_  = get_top_level_operands(args_tokens,terminators=[])
    if len(tokens):
        raise error.SyntaxError("could not parse CUDA Fortran kernel call: trailing text")
    return kernel_name, params, args

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
        try:
            label = str(int(var))
            var = tokens.pop(0)
        except ValueError as e:
            raise error.SyntaxError("expected identifier or numeric label after 'do'") from e
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
        stride = range_vals[2]
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
            if len(parts) > 1:
                varname = "%".join(parts[:-1]+[varname])
            if all_tokens_are_blank(child_tokens):
                var_tuple = (varname, [])
                result1.append(var_tuple)
            elif child_tokens[0] == "(":
                child_tokens.pop(0)
                ranges,_  = get_top_level_operands(child_tokens) # : ["-1:m", "n"]
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
            else:
                raise error.SyntaxError("expected end of allocate argument or '('")
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

def parse_lvalue(statement):
    tokens = tokenize(statement)
    parts,_ = get_top_level_operands(tokens,separators=["%"],
                    join_operand_tokens=False)
    result = []
    for part_tokens in parts:
        ident = part_tokens.pop(0)
        if not ident.isidentifier():
            raise error.SyntaxError("expected identifier")
        if len(part_tokens) and part_tokens[0] == "(":
            part_tokens.pop(0)
            args,num_argument_tokens = get_top_level_operands(part_tokens)
            part_tokens = part_tokens[num_argument_tokens:]
            if len(part_tokens) and part_tokens.pop(0) != ")":
                raise error.SyntaxError("expected ')'")
            if not len(args):
                raise error.SyntaxError("expected at least one argument")
        elif len(part_tokens):
            raise error.SyntaxError("expected '(' or no token")
        else:
            args = []
        result.append((ident,args))
    return result

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

def parse_parameter_statement(statement):
    """:return: List of tuples consisting of parameter name and assigned value.
    ```
    PARAMETER( a = 5, b = 3)
    PARAMETER b = 5, b = 3   ! legacy version
    ```
    will both result in :
    [('a','5'),('b','3')]
    """
    tokens = tokenize(statement)
    result = []
    if tokens.pop(0).lower() != "parameter":
        raise error.SyntaxError("expected 'parameter'")
    if tokens[0] == "(":
        # modern variant
        tokens.pop(0)
        variables_raw, num_consumed_tokens  = get_top_level_operands(tokens)
        tokens = tokens[num_consumed_tokens:]
        if tokens.pop(0) != ")":
            raise error.SyntaxError("expected ')'")
    elif tokens[0].isidentifier():
        variables_raw, num_consumed_tokens  = get_top_level_operands(tokens)
        tokens = tokens[num_consumed_tokens:]
    else:
        raise error.SyntaxError("expected '(' or identifier")
    check_if_all_tokens_are_blank(tokens)
    if not len(variables_raw):
        raise error.SyntaxError("expected at least one assignment expression")
    for expr in variables_raw: 
        lhs,rhs = parse_assignment(expr)
        if not lhs.isidentifier():
            raise error.SyntaxError("expected identifier")
        result.append((lhs,rhs))
    return result

# rules
def is_declaration(tokens):
    """No 'function' must be in the tokens.
    """
    return (tokens[0].lower() in ["type","integer","real","complex","logical","character"]
           or compare_ignore_case(tokens[0:2],["double","precision"]))

def is_blank_line(statement):
    return not len(statement.strip())


def is_fortran_directive(statement,modern_fortran):
    """If the statement is a directive."""
    return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:2] == "!$" 
       or (not modern_fortran and statement[0:2].lower() in ["c$","*$"]))

def is_cuda_fortran_conditional_code(statement,modern_fortran):
    """If the statement is a directive."""
    return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:5].lower() == "!@cuf" 
       or (not modern_fortran and statement[0:5].lower() in ["c@cuf","*@cuf"]))

def is_fortran_comment(statement,modern_fortran):
    """If the statement is a directive.
    :note: Assumes that `is_cuda_fortran_conditional_code` and 
           `is_fortran_directive` are checked before.
    """
    return len(statement) and (modern_fortran and statement.lstrip()[0] == "!" 
       or (not modern_fortran and statement[0].lower() in "c*"))


def is_cpp_directive(statement):
    return statement[0] == "#"


def is_fortran_offload_region_directive(tokens):
    return\
        compare_ignore_case(tokens[1:3],["acc","serial"]) or\
        compare_ignore_case(tokens[1:3],["acc","parallel"]) or\
        compare_ignore_case(tokens[1:3],["acc","kernels"])

def is_fortran_offload_region_plus_loop_directive(tokens):
    return\
        compare_ignore_case(tokens[1:4],["cuf","kernel","do"])  or\
        compare_ignore_case(tokens[1:4],["acc","parallel","loop"]) or\
        compare_ignore_case(tokens[1:4],["acc","kernels","loop"]) or\
        compare_ignore_case(tokens[1:4],["acc","kernels","loop"])

def is_fortran_offload_loop_directive(tokens):
    return compare_ignore_case(tokens[1:3],["acc","loop"])

def is_derived_type_member_access(tokens):
    return len(get_top_level_operands(tokens,separators=["%"])[0]) > 1

def is_assignment(tokens):
    """:return if the statement tokens can be split into two
               parts w.r.t. "=" and the left-hand operand
               is an lvalue expression.
    """
    #assert not is_declaration_(tokens)
    #assert not is_do_(tokens)
    if "=" in tokens:
        operands,_ = get_top_level_operands(tokens,separators=["="])
        if len(operands) == 2:
            lhs = operands[0]
            try:
                parse_lvalue(lhs)
                return True
            except:
                return False
        else:
            return False
    else:
        return False

def is_pointer_assignment(tokens):
    #assert not is_ignored_statement_(tokens)
    #assert not is_declaration_(tokens)
    return len(get_top_level_operands(tokens,separators=["=>"])[0]) == 2

def is_subroutine_call(tokens):
    return tokens[0] == "call" and tokens[1].isidentifier()

def is_select_case(tokens):
    cond1 = compare_ignore_case(tokens[0:2],["select", "case"])
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            compare_ignore_case(tokens[2:4],["select","case"])
    return cond1 or cond2

def is_case(tokens):
    return compare_ignore_case(tokens[0:2],["case", "("])


def is_case_default(tokens):
    return compare_ignore_case(tokens[0:2],["case", "default"])

def is_if_then(tokens):
    """:note: we assume single-line if have been
    transformed in preprocessing step."""
    return compare_ignore_case(tokens[0:2],["if", "("])

def is_else_if_then(tokens):
    return compare_ignore_case(tokens[0:3],["else", "if", "("])

def is_else(tokens):
    #assert not is_else_if_then_(tokens)
    return tokens[0] == "else"

def is_do_while(tokens):
    cond1 = compare_ignore_case(tokens[0:2],["do", "while"])
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            compare_ignore_case(tokens[2:4],["do","while"])
    return cond1 or cond2

def is_do(tokens):
    cond1 = tokens[0] == "do"
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2] == "do"
    return cond1 or cond2

def is_continue(tokens):
    if tokens[0].lower() == "continue":
        return True
    elif tokens[1].lower() == "continue":
        try:
            int(tokens[0])
            return True
        except:
            return False 

def is_where(tokens):
    cond1 = tokens[0].lower() == "where"
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2].lower() == "where"
    return cond1 or cond2

def is_end(tokens, kinds=[]):
    cond1 = tokens[0].lower() == "end"
    cond2 = not len(kinds) or tokens[1].lower() in kinds
    return cond1 and cond2
