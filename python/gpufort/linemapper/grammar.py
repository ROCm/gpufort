# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing
import re

pyparsing.ParserElement.enablePackrat()

## INGREDIENTS
LPAR,RPAR  = map(pyparsing.Suppress, "()")

pp_ident      = pyparsing.pyparsing_common.identifier.copy()
pp_number     = pyparsing.pyparsing_common.number.copy().setParseAction(lambda tk: str(tk[0]))
pp_bool_true  = pyparsing.Regex(r"\.true\.|true|1",re.IGNORECASE).setParseAction(lambda tk: "1")
pp_bool_false = pyparsing.Regex(r"\.false\.|false|0",re.IGNORECASE).setParseAction(lambda tk: "0")
pp_defined    = pyparsing.Regex(r"defined\(\s*(?P<name>\w+)\s*\)",re.IGNORECASE)

pp_value = pyparsing.Forward()

pp_op_and = pyparsing.Regex(r"\.and\.|&&",re.IGNORECASE).setParseAction(lambda tk: "and")
pp_op_or  = pyparsing.Regex(r"\.or\.|\|\|",re.IGNORECASE).setParseAction(lambda tk: "or")
pp_op_not = pyparsing.Regex(r"\.not\.|!",re.IGNORECASE).setParseAction(lambda tk: "not ")

# only needed for macro args because we hand conditions to the python ast.literal_eval function
# after performing some transformations
# https://en.cppreference.com/w/c/language/operator_precedence
pp_arithm_logic_expr =  pyparsing.infixNotation(pp_value, [
    (pp_op_not,               1, pyparsing.opAssoc.RIGHT),
    (pyparsing.Regex(r"\*|/|%"),    2, pyparsing.opAssoc.LEFT),
    (pyparsing.Regex(r"\+|-"),      2, pyparsing.opAssoc.LEFT),
    (pyparsing.Regex(r"<=|>=|<|>"), 2, pyparsing.opAssoc.LEFT),
    (pyparsing.Regex(r"==|!="),     2, pyparsing.opAssoc.LEFT),
    (pp_op_and,               2, pyparsing.opAssoc.LEFT),
    (pp_op_or,                2, pyparsing.opAssoc.LEFT)
])

arg_list = (LPAR + RPAR).setParseAction(lambda tk: []) | \
  (LPAR + pyparsing.delimitedList( pp_arithm_logic_expr ) + RPAR)
pp_macro_eval   = pp_ident + pyparsing.Group(pyparsing.Optional(arg_list,default=[])) # ! setResultName causes issue when args are macro evals again
pp_value      <<= ( pp_number | pp_bool_true | pp_bool_false | pp_defined | pp_macro_eval ) # | pp_char ) # ! defined must come before eval

# DIRECTIVES

# if
pp_dir_ifdef   = pyparsing.Regex(r"#\s*ifdef\s+(?P<name>\w+)",re.IGNORECASE)
pp_dir_ifndef  = pyparsing.Regex(r"#\s*ifndef\s+(?P<name>\w+)",re.IGNORECASE)
pp_dir_if      = pyparsing.Regex(r"#\s*if\s+(?P<condition>.+)",re.IGNORECASE)
# elif
pp_dir_elif    = pyparsing.Regex(r"#\s*elif\s+(?P<condition>.+)",re.IGNORECASE)
# else
pp_dir_else    = pyparsing.Regex(r"#\s*else\b",re.IGNORECASE)
# include
pp_dir_include = pyparsing.Regex(r"#\s*include\s+\"(?P<filename>["+pyparsing.printables+r"]+)\"",re.IGNORECASE)
# define / undef
pp_dir_define  = pyparsing.Regex(r"#\s*define\s+(?P<name>\w+)(\((?P<args>(\w|[, ])+)\))?\s+(?P<subst>.*)",re.IGNORECASE)
pp_dir_undef   = pyparsing.Regex(r"#\s*undef\s+(?P<name>\w+)",re.IGNORECASE)
# other
pp_compiler_option = pyparsing.Regex(r"-D(?P<name>\w+)(=(?P<value>["+pyparsing.printables+r"]+))?")

pp_ops = pp_op_and | pp_op_or | pp_op_not