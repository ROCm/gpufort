# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

import pyparsing

from . import parsing

pp_defined = pyparsing.Regex(r"defined\(\s*(?P<name>\w+)\s*\)")

pp_compiler_option = pyparsing.Regex(r"-D(?P<name>\w+)(=(?P<value>["
                                     + pyparsing.printables + r"]+))?")

def init_cpp_macros(options):
    """init macro stack from C compiler options and user-prescribed config values."""
    macro_stack = []
    for result, _, __ in pp_compiler_option.scanString(options):
        value = result.value
        if value == None:
            value == "1"
        macro = ( result.name, [], result.value )
        macro_stack.append(macro)
    return macro_stack

def _evaluate_defined(input_string, macro_names):
    # expand macro; one at a time
    result = input_string
    iterate = True
    while iterate:
        iterate = False
        for match, start, end in pp_defined.scanString(result):
            substring = result[start:end].strip(" \t\n")
            subst = "1" if match.name in macro_names else "0"
            result = result.replace(substring, subst)
            iterate = True
            break
    return result

def expand_macros(input_string, macro_stack, is_cpp_directive=False, ignore_case=False):
    """Recursively expand expressions in the input string
    by substitutions described by the macro stack.
    :param bool is_cpp_directive: If the input_string is a C preprocessor
                                  directive. In this case, also expand `defined( <varname> )`
                                  expressions. Defaults to False.
    :param bool ignore_case: If the case of the macro names and parameters should be ignored
                             when expanding macros. Defaults to False. If `is_cpp_directive` is set, this parameter
                             has no effect as C preproc. directives are case sensitive.
    """
    # expand macro; one at a time
    if is_cpp_directive:
        ignore_case = False
    macro_names = [macro[0] for macro in macro_stack]
    # check if macro identifiers appear in string before scanning the result
    iterate = True
    tokens = parsing.tokenize(input_string)
    tokens_lower = parsing.tokenize(input_string.lower())
    for name in macro_names:
        if name in tokens or (ignore_case and name.lower() in tokens_lower):
            iterate = True
    if is_cpp_directive and (iterate or "defined" in tokens):
        result = _evaluate_defined(input_string, macro_names)
    else:
        result = input_string
    while iterate:
        # Requires N+1 iterations to perform N replacements
        iterate = False
        for macro in macro_stack:
            name, args, subst = macro
            if subst == None:
                subst = ""
            subst = subst.strip(" \n\t")
            result_tokens = parsing.tokenize(result)
            result_tokens_lower = parsing.tokenize(result.lower())
            if (name in result_tokens
               or (ignore_case 
                   and name in result_tokens_lower)):
                if len(args):
                    calls = parsing.extract_function_calls(result, name)
                    if len(calls):
                        substring = calls[-1][
                            0] # always start from right-most (=inner-most) call site
                        params = calls[-1][1]
                        for n, placeholder in enumerate(args):
                            if ignore_case:
                                subst = re.sub(r"\b{}\b".format(placeholder),
                                               params[n], subst, re.IGNORECASE)
                            else:
                                subst = re.sub(r"\b{}\b".format(placeholder),
                                               params[n], subst)
                        result = result.replace(substring,
                                                subst) # result has changed
                        iterate = True
                        break
                else:
                    old_result = result
                    if ignore_case:
                        result = re.sub(r"\b{}\b".format(name), subst, result, re.IGNORECASE)
                    else:
                        result = re.sub(r"\b{}\b".format(name), subst, result)
                    if result != old_result:
                        iterate = True
                    break
    return result

def _convert_operators(input_string):
    """Converts Fortran and C logical operators to python ones.
    :note: Adds whitespace around the operators as they 
    """
    return input_string.replace(r".and."," and ").\
                        replace(r".or."," or ").\
                        replace(r".not.","not ").\
                        replace(r"&&"," and ").\
                        replace(r"||"," or ").\
                        replace(r"!","not ")


def evaluate_condition(input_string, macro_stack):
    """
    Evaluates preprocessor condition.
    :param str input_string: Expression as text.
    :note: Input validation performed according to:
           https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval
    """
    transformed_input_string = _convert_operators(
        expand_macros(input_string, macro_stack, True))
    code = compile(transformed_input_string, "<string>", "eval")
    return eval(code, {"__builtins__": {}}, {}) > 0
