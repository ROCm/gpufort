# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

from .. import error


def compare(tokens1, # type: Union[str,list,tuple]
            tokens2): # type: Union[str,list,tuple]
    if isinstance(tokens1,str):
        if isinstance(tokens2,str):
            return tokens1 == tokens2
        else:
            return tokens1 == tokens2[0]
    else:
        if len(tokens1) != len(tokens2):
            return False
        else:
            return next(
              (tk for i,tk in enumerate(tokens1) 
              if tk != tokens2[i]),None) == None

def compare_ignore_case(tokens1, # type: Union[str,list,tuple]
                        tokens2): # type: Union[str,list,tuple]
    if isinstance(tokens1,str):
        if isinstance(tokens2,str):
            return tokens1.lower() == tokens2.lower()
        else:
            return tokens1.lower() == tokens2[0].lower()
    else:
        if len(tokens1) != len(tokens2):
            return False
        else:
            return next(
              (tk for i,tk in enumerate(tokens1) 
              if tk.lower() != tokens2[i].lower()),None) == None

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
        return list(tokens)

def tokenize(statement, padded_size=0, modern_fortran=True,keepws=False):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param statement: String or list of strings.
    :param int padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    :param bool keepws: keep whitespaces in the resulting list of tokens.
    """
    if isinstance(statement,str):
        TOKENS_KEEP = [
          r"[\"](?:\\.|[^\"\\])*[\"]",
          r"[\'](?:\\.|[^\'\\])*[\']",
          r"::?",
          r"<<<",
          r">>>",
          r"[<>]=?",
          r"[\/=]=",
          r"=>?",
          r"\*\*?",
          r"\.(?:[a-zA-Z]+)\.", # operators cannot have "_"
          r"[();,%+-/&]",
          r"[\s\t\r\n]+",
        ]
        if modern_fortran:
            TOKENS_KEEP.append(r"![@\$]?")
        else:
            TOKENS_KEEP.append(r"^[cC\*dD][@\$]?")
        # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
        # is not captured.
        keep_pattern = "(" + "|".join(TOKENS_KEEP) + ")"
        tokens = re.split(keep_pattern, statement, 0) # re.IGNORECASE
        # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
        # is not captured.
        if keepws: 
            # we need to remove zero-length tokens, todo: not sure why they appear
            return pad_to_size([tk for tk in tokens if len(tk)],padded_size)
        else:
            return pad_to_size([tk for tk in tokens if len(tk) and not tk.isspace()],padded_size)
    elif isinstance(statement,list):
        if keepws:
            return pad_to_size(statement,padded_size)
        else:
            return pad_to_size([tk for tk in statement if len(tk.strip())],padded_size)
    else:
        raise Exception("input must be either a str or a list of strings")

def remove_whitespace(statement):
    """Remove whitespace.
    :param statement: String or list of strings.
    """
    if isinstance(statement,list):
        return tokenize(statement)
    else:
        return "".join(tokenize(statement))


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

    :param str var_expr: a simple identifier such as 'a' or 'A_d', an array section expression such as 'B(i,j)' or a more complicated derived-type member variable expression such as 'a%b%c' or 'A%b(i)%c'.
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

def is_blank(text):
    return not len(text) or text.isspace()

def split_fortran_line(line,modern_fortran=True):
    """Decomposes a Fortran line into numeric label, preceding whitespace,
     statement part, comment part, and trailing whitespace (including newline char).
     """
    tokens = tokenize(line,modern_fortran=modern_fortran,keepws=True)
    # numeric_label
    numeric_label_part = ""
    if ( len(tokens) >= 2 # numeric label cannot stand alone
         and tokens[0].isnumeric()):
          numeric_label_part = tokens[0] 
          tokens = tokens[1:]
    # whitespace (can be in front of named label)
    if len(tokens) and is_blank(tokens[0]):
        preceding_ws = tokens.pop(0)
    else: 
        preceding_ws = ""
    if len(tokens) and is_blank(tokens[-1]):
        trailing_ws = tokens.pop(-1)
    else: 
        trailing_ws = ""
    # collect statement or directive tokens and comment tokens
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
    return ( 
      numeric_label_part, 
      preceding_ws, 
      "".join(statement_tokens), 
      "".join(comment_tokens),
      trailing_ws
    )

def detect_line_starts(lines,modern_fortran=True):
    """Fortran statements can be broken into multiple lines 
    via the '&' characters. This routine detects in which line a statement
    (or multiple statements per line) begins.
    The difference between the line numbers of consecutive entries
    is the number of lines the first statement occupies.
    """
    # 1. save multi-line statements (&) in buffer
    buffering = False
    line_starts = []
    for lineno, line in enumerate(lines, start=0):
        # Continue buffering if multiline CUF/ACC/OMP statement
        _,_,stmt_or_dir,comment,_ = split_fortran_line(line)
        stmt_or_dir_tokens = tokenize(stmt_or_dir)
        # continuation at line begin
        if (len(stmt_or_dir_tokens) > 3
           and (modern_fortran and stmt_or_dir_tokens[0] in ["!$","!@"]
               or not modern_fortran and stmt_or_dir_tokens[0].lower() in ["c$","c@","*$","*@"])):
            buffering = buffering or stmt_or_dir_tokens[2] == "&" 
        if not buffering:
            line_starts.append(lineno)
        if len(stmt_or_dir_tokens) and stmt_or_dir_tokens[-1] in ['&', '\\']:
            buffering = True
        elif len(stmt_or_dir_tokens):
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
        numeric_label_part,indent,stmt_or_dir,comment,trailing_ws = \
          split_fortran_line(line,modern_fortran)
        # note: label_part might actually only be 
        # a label in the first line of a multiline statement.
        # In other lines, it can be a continuation of
        # of an operation that puts an integer number 
        # at column 0. 
        stmt_or_dir_tokens = tokenize(stmt_or_dir)
        if len(stmt_or_dir_tokens) and stmt_or_dir_tokens[-1] == "&":
            in_multiline_statement = True
        if in_multiline_statement and len(comment):
            if len(stmt_or_dir):
                result.append(
                  numeric_label_part + indent + stmt_or_dir + trailing_ws)
            comment_buffer.append(indent + comment + trailing_ws)
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
    # todo: throw error if open_brackets still > 0
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
    num_consumed = 0
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
        num_consumed += 1
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
            num_consumed -= 1
        else: 
            current_operand.append(tk)
    if len(current_operand):
        if join_operand_tokens:
            result.append(_join_tokens(current_operand,no_ws))
        else:
            result.append(current_operand)
    return result, num_consumed

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