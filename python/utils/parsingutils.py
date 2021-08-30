import re

def tokenize(statement,padded_size=0):
    """Splits string at whitespaces and substrings such as
    'end', '$', '!', '(', ')', '=>', '=', ',' and "::".
    Preserves the substrings in the resulting token stream but not the whitespaces.
    :param str padded_size: Always ensure that the list has at least this size by adding padded_size-N empty
               strings at the end of the returned token stream. Has no effect if N >= padded_size. 
               Disable padding by specifying value <= 0.
    """
    TOKENS_REMOVE = r"\s+|\t+"
    TOKENS_KEEP   = r"(end|else|!\$?|(c|\*)\$|[(),]|::?|=>?|<<<|>>>|(<|>)=?|(/|=)=|\+|-|\*|/|(\.\w+\.))"
    
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
