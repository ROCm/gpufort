import re

COMMENT_CHARS="!cCdD*"

def split_fortran_line(line):
     global COMMENT_CHARS

     """Decomposes a Fortran line into preceding whitespace,
     statement part, comment part, and trailing whitespace (including newline char).
     """
     len_line         = len(line)
     len_preceding_ws = len_line-len(line.lstrip(" \n\t"))
     len_trailing_ws  = len_line-len(line.rstrip(" \n\t"))
     content          = line[len_preceding_ws:len_line-len_trailing_ws]

     statement = ""
     comment   = ""
     if len(content):
         if len_preceding_ws == 0 and content[0] in COMMENT_CHARS or\
            content[0] == "!":
             if len(content) > 1 and content[1] == "$":
                 statement = content
             else:
                 comment   = content
         else:
             content_parts = content.split("!",maxsplit=1)
             if len(content_parts) > 0:
                 statement = content_parts[0]
             if len(content_parts) == 2:
                 comment = "!"+content_parts[1]
     preceding_ws = line[0:len_preceding_ws]
     trailing_ws  = line[len_line-len_trailing_ws:]
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
    result         = []
    comment_buffer = []
    in_multiline_statement = False
    for line in lines:
       indent,stmt_or_dir,comment,trailing_ws = \
         split_fortran_line(line)
       if len(stmt_or_dir) and stmt_or_dir[-1] == "&":
           in_multiline_statement = True
       if in_multiline_statement and len(comment):
           if len(stmt_or_dir):
               result.append("".join([indent,stmt_or_dir,trailing_ws]))
           comment_buffer.append("".join([indent,comment,trailing_ws]))
       else:
           result.append(line)
       if len(stmt_or_dir) and stmt_or_dir[-1] != "&":
           result += comment_buffer
           comment_buffer.clear()
           in_multiline_statement = False 
    return result

def tokenize(statement,padded_size=0):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    """
    TOKENS_REMOVE = r"\s+|\t+"
    TOKENS_KEEP   = r"(end|else|!\$?|[c\*]\$|[(),]|::?|=>?|<<<|>>>|[<>]=?|[/=]=|\+|-|\*|/|\.\w+\.)"
    # IMPORTANT: Use non-capturing groups (?:<expr>) to ensure that an inner group in TOKENS_KEEP
    # is not captured.
    
    tokens1 = re.split(TOKENS_REMOVE,statement)
    tokens  = []
    for tk in tokens1:
        tokens += [part for part in re.split(TOKENS_KEEP,tk,0,re.IGNORECASE)]
    result = [tk for tk in tokens if tk != None and len(tk.strip())]
    if padded_size > 0 and len(result) < padded_size:
        return result + [""]*(padded_size-len(result))
    else:
        return result

def next_tokens_till_open_bracket_is_closed(tokens,open_brackets=0):
    # ex:
    # input:  [  "kind","=","2","*","(","5","+","1",")",")",",","pointer",",","allocatable" ], open_brackets=1
    # result: [  "kind","=","2","*","(","5","+","1",")",")" ]
    result    = []
    idx       = 0
    criterion = True
    while criterion:
        tk   = tokens[idx]
        result.append(tk)
        idx += 1
        if tk == "(":
            open_brackets += 1
        elif tk == ")":
            open_brackets -= 1
        criterion = idx < len(tokens) and open_brackets > 0
    return result

def create_comma_separated_list(tokens,open_brackets=0,separators=[","],terminators=["::","\n","!"]):
    # ex:
    # input: ["parameter",",","intent","(","inout",")",",","dimension","(",":",",",":",")","::"]
    # result : ["parameter", "intent(inout)", "dimension(:,:)" ]
    result            = []
    idx               = 0
    current_qualifier = ""
    criterion         = len(tokens)
    while criterion:
        tk  = tokens[idx]
        idx += 1
        criterion = idx < len(tokens)
        if tk in separators and open_brackets == 0:
            if len(current_qualifier):
                result.append(current_qualifier)
            current_qualifier = ""
        elif tk in terminators:
            criterion = False
        else:
            current_qualifier += tk
        if tk == "(":
            open_brackets += 1
        elif tk == ")":
            open_brackets -= 1
    if len(current_qualifier):
        result.append(current_qualifier)
    return result

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
def is_comment(tokens,statement):
    cond1 = tokens[0]=="!"
    cond2 = len(statement) and statement[0] in ["c","*"]
    return cond1 or cond2
def is_cpp_directive(statement):
    return statement[0] == "#" 
def is_fortran_directive(tokens,statement):
    return tokens[0] in ["!$","*$","c$"]
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
    cond1 = tokens[0:1+1] == ["select","case"]
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2:3+1] == ["select","case"]
    return cond1 or cond2
def is_case(tokens):
    return tokens[0:1+1] == ["case","("]
def is_case_default(tokens):
    return tokens[0:1+1] == ["case","default"]
def is_if_then(tokens):
    """:note: we assume single-line if have been
    transformed in preprocessing step."""
    return tokens[0:1+1] == ["if","("]
def is_if_then(tokens):
    """:note: we assume single-line if have been
    transformed in preprocessing step."""
    return tokens[0:1+1] == ["if","("]
def is_else_if_then(tokens):
    return tokens[0:2+1] == ["else","if","("]
def is_else(tokens):
    #assert not is_else_if_then_(tokens)
    return tokens[0] == "else"
def is_do_while(tokens):
    cond1 = tokens[0:1+1] == ["do","while"]
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2:3+1] == ["do","while"]
    return cond1 or cond2
def is_do(tokens):
    cond1 = tokens[0] == "do"
    cond2 = tokens[0].isidentifier() and tokens[1] == ":" and\
            tokens[2] == "do"
    return cond1 or cond2
def is_end(tokens,kinds=[]):
    cond1 = tokens[0] == "end"
    cond2 = not len(kinds) or tokens[1] in kinds
    return cond1 and cond2
