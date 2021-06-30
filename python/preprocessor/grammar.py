import pyparsing as pyp
import re

## INGREDIENTS

pp_ident           = pyp.pyparsing_common.identifier.copy()
LPAR,RPAR          = map(pyp.Suppress, "()")

# conditions
#pp_char             = pyp.Regex(r"'[ -~]'")
pp_number            = pyp.pyparsing_common.number.copy().setParseAction(lambda tk: str(tk))
pp_bool_true         = pyp.Regex(r"\.true\.|true|1",re.IGNORECASE).setParseAction(lambda tk: "1")
pp_bool_false        = pyp.Regex(r"\.false\.|false|0",re.IGNORECASE).setParseAction(lambda tk: "0")
pp_defined           = pyp.Regex(r"defined\s*\(\s*(?P<key>\w+)\s*\)",re.IGNORECASE)

pp_value = pyp.Forward()

pp_op_and = pyp.Regex(r"\.and\.|&&",re.IGNORECASE).setParseAction(lambda tk: "and" )
pp_op_or  = pyp.Regex(r"\.or\.|\|\|",re.IGNORECASE).setParseAction(lambda tk: "or") 
pp_op_not = pyp.Regex(r"\.not\.|!",re.IGNORECASE).setParseAction(lambda tk: "not")

# only needed for macro args because we hand conditions to the python ast.literal_eval function
# after performing some transformations
# https://en.cppreference.com/w/c/language/operator_precedence
pp_arithm_logic_expr =  pyp.infixNotation(pp_value, [
    (pp_op_not,               1, pyp.opAssoc.RIGHT),
    (pyp.Regex(r"\*|/"),      2, pyp.opAssoc.LEFT),
    (pyp.Regex(r"\+|-"),      2, pyp.opAssoc.LEFT),
    (pyp.Regex(r"<|>|<=|>="), 2, pyp.opAssoc.LEFT),
    (pyp.Regex(r"==|!="),     2, pyp.opAssoc.LEFT),
    (pp_op_and,               2, pyp.opAssoc.LEFT),
    (pp_op_or,                2, pyp.opAssoc.LEFT)
])

pp_ident_or_macro_eval = pp_ident + pyp.Optional(LPAR + pyp.delimitedList( pp_arithm_logic_expr ) + RPAR,default=[])
pp_value      <<= ( pp_number | pp_bool_true | pp_bool_false | pp_defined | pp_ident_or_macro_eval ) # | pp_char )

# DIRECTIVES

# if
pp_dir_ifdef   = pyp.Regex(r"#\s*ifdef\s+(?P<name>\w+)",re.IGNORECASE)
pp_dir_ifndef  = pyp.Regex(r"#\s*ifndef\s+(?P<name>\w+)",re.IGNORECASE)
pp_dir_if      = pyp.Regex(r"#\s*if\s+(?P<condition>.+)$",re.IGNORECASE)
# elif
pp_dir_elif    = pyp.Regex(r"#\s*elif\s+(?P<condition>.+)",re.IGNORECASE)
# else
pp_dir_else    = pyp.Regex(r"#\s*else\b",re.IGNORECASE)
# include
pp_dir_include = pyp.Regex(r"#\s*include\s+\"(?P<filename>["+pyp.printables+r"\"]+)",re.IGNORECASE)
# define / undef
pp_dir_define  = pyp.Regex(r"#\s*define\s+(?P<name>\w+)",re.IGNORECASE) +\
                 pyp.Optional(LPAR + pyp.Optional(pyp.delimitedList(pp_ident),default=[]) + RPAR,default=[]).setResultsName("args") +\
                 pyp.Regex(".*$").setResultsName("subst")
pp_dir_undef   = pyp.Regex(r"#\s*undef\s+(?P<name>\w+)",re.IGNORECASE)
# other
pp_compiler_option = pyp.Regex(r"-D(?P<name>\w+)(=(?P<value>["+pyp.printables+r"]+))?")
