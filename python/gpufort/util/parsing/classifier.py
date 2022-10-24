#  SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import error

from . import base

from . import acc_clauses
from . import tokenstream as ts
    
_basic_types = ["logical","integer","real","complex"]

class StatementClassifier:

    # decorators
    def __produces_parse_result(func):
        def inner(self,*args,**kwargs):
            passed = func(self,*args,**kwargs)
            self.__result_avail = passed
            return passed
        return inner
    
    def __no_parse_result(func):
        def inner(self,*args,**kwargs):
            self.__result_avail = False
            return func(self,*args,**kwargs)
        return inner
    
    def __init__(self):
        self.__parse_result = None
        self.__result_avail = False
    
    def _raise_unclassified_error(self,tokens,msg):
        """Raise error when the statement has not been classified yet."""
        #todo: Parametrize error object to tell exception catching
        # caller that this statement has not been classified yet.
        raise error.SyntaxError(msg)

    def _raise_classified_error(self,tokens,msg):
        """Raise error when the statement has been classified."""
        #todo: Parametrize error object to tell exception catching
        # caller that this statement has been classified already.
        raise error.SyntaxError(msg)
    
    @property
    def parse_result_is_available(self):
        return self.__result_avail        
    @property
    def parse_result(self):
        assert self.__result_avail
        return self.__parse_result 
   

    def _parse_attributes_list(self,tokens):
        if not tokens.pop_front_equals_ignore_case("attributes","("):
            self._raise_unclassified_error(tokens,"expected 'attributes' + '('")
        attributes, num_consumed =  base.get_top_level_operands(tokens)
        if not len(attributes):
            raise self._raise_unclassified_error(tokens,"expected at least one attribute in the 'attributes' list")
        tokens.pop_front(num_consumed)
        if not tokens.pop_front_equals(")"):
            raise self._raise_unclassified_error(tokens,"expected ')' after 'attributes' list")
        return attributes
 
    def parse_interface_statement(self,statement):
        """Parse interface statements such as
        `interface`
        `INTERFACE myinterface`
        `attributes(std_f77) INTERFACE myinterface`
        The latter expression is a non-standard extension.
        """
        tokens = ts.TokenStream(statement)
        if tokens[0].lower() == "attributes":
            attributes = self._parse_attributes_list(tokens)
        else:
            attributes = []
        if not tokens.pop_front_equals_ignore_case("interface"):
            self._raise_unclassified_error(tokens,"expected 'interface'")
        if len(tokens) and tokens[0].isidentifier():
            name = tokens.pop_front()
        else:
            name = None
        tokens.check_if_remaining_tokens_are_blank()
        return (attributes, name)
    
    def parse_function_statement(self,statement):
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
    
        tokens = base.tokenize(statement,padded_size=10)
        criterion = len(tokens)
        # prefix
        while criterion:
            if tokens[0].lower() in type_begin:
                result_type, f_len, result_type_kind, params, num_consumed =\
                        self.parse_fortran_datatype(tokens)
                tokens=tokens[num_consumed:]
            else:
                tk = tokens.pop(0)
                criterion = len(tokens)
                if tk.lower() in simple_modifiers:
                    modifiers.append(tk)
                elif tk.lower() == "attributes":
                    if not tokens.pop(0) == "(":
                        raise error.SyntaxError("expected '(' after 'attributes'")
                    attributes, num_consumed =  base.get_top_level_operands(tokens)
                    if not len(attributes):
                        raise error.SyntaxError("expected at least one attribute in the 'attributes' list")
                    tokens = tokens[num_consumed:]
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
                dummy_args, num_consumed = base.get_top_level_operands(tokens)  
                tokens = tokens[num_consumed:]
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
                operands, num_consumed =  base.get_top_level_operands(tokens)
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
                tokens = tokens[num_consumed:]
                if not tokens.pop(0) == ")":
                    raise error.SyntaxError("expected ')' after 'attributes' list")
            elif len(tk.strip()):
                raise error.SyntaxError("unexpected suffix token: {}".format(tk))
            criterion = len(tokens)
        base.check_if_all_tokens_are_blank(tokens)
        return (
            kind, name, dummy_args, modifiers, attributes,
            (result_type, result_type_kind, result_name),
            (bind_c, bind_c_name)
        )
    
    def parse_use_statement(self,statement):
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
        tokens = base.tokenize(statement,10)
        if tokens.pop(0).lower() != "use":
            raise error.SyntaxError("expected 'use'")
        module_name = None
        qualifiers  = []
        tk = tokens.pop(0)
        if tk.isidentifier():
            module_name = tk
        elif tk == ",":
            qualifiers,num_consumed = base.get_top_level_operands(tokens)
            tokens = tokens[num_consumed:]
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
           and base.compare_ignore_case(tokens[0:3],[",","only",":"])):
            tokens = tokens[3:]
            have_only = True
        elif (len(tokens) > 4
             and tokens[0] == ","):
            tokens.pop(0)
            have_renamings = True
        if have_only or have_renamings:
            entries,num_consumed = base.get_top_level_operands(tokens,join_operand_tokens=False)
            tokens = tokens[num_consumed:]
            for entry in entries:
                parts,_ = base.get_top_level_operands(entry,separators=["=>"])
                if len(parts) == 1 and have_only:
                    mappings.append((parts[0],parts[0]))  # use-name, use-name
                elif len(parts) == 2:
                    mappings.append((parts[0], parts[1])) # local-name, use-name
                elif have_only:
                    raise error.SyntaxError("unexpected only list entry: {}".format(entry))
                elif have_renamings:
                    raise error.SyntaxError("unexpected rename list entry: {}".format(entry))
        if have_only:
            only = mappings
            renamings = []
        else:
            only = []
            renamings = mappings
        base.check_if_all_tokens_are_blank(tokens)
        return module_name, qualifiers, renamings, only
    
    def parse_attributes_statement(self,statement):
        """Parses a CUDA Fortran attributes statement.
        :return: A tuple with the attribute as first
                and the subjected variables (list of str)
                as second entry.
       
        :Example:
       
        `attributes(device) :: var_a, var_b`
       
        will result in 
    
        ('device', ['var_a', 'var_b'])
        """
        tokens = base.tokenize(statement,10)
    
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
    
    def _parse_raw_qualifiers_and_rhs(self,tokens):
        """:return: Triple of qualifiers and variables as list of raw strings plus
                    the number of consumed tokens (in that order).
        """
        DOUBLE_COLON = "::"
        try:
            qualifiers_raw = []
            idx_last_consumed_token = None
            if tokens[0] == "," and DOUBLE_COLON in tokens: # qualifier list
                qualifiers_raw,_  = base.get_top_level_operands(tokens)
                idx_last_consumed_token = tokens.index(DOUBLE_COLON)
            elif tokens[0] == DOUBLE_COLON: # variables begin afterwards
                idx_last_consumed_token = 0
            elif tokens[0].isidentifier(): # variables begin directly
                idx_last_consumed_token = -1
            else:
                raise error.SyntaxError("could not parse qualifiers and right-hand side")
            # handle variables list
            num_consumed = idx_last_consumed_token+1
            tokens = tokens[num_consumed:]
            variables_raw, num_consumed2  = base.get_top_level_operands(tokens)
            num_consumed += num_consumed2
            if not len(variables_raw):
                raise error.SyntaxError("expected at least one entry in right-hand side")
            return qualifiers_raw, variables_raw, num_consumed
        except IndexError:
            raise error.SyntaxError("could not parse qualifiers and right-hand side")
    
    def parse_type_statement(self,statement):
        """Parse a Fortran type statement.
        :return: List 
        """
        tokens = base.tokenize(statement,10)
        if tokens.pop(0).lower() != "type":
            raise error.SyntaxError("expected 'type'")
        attributes, rhs, num_consumed = self._parse_raw_qualifiers_and_rhs(tokens)
        tokens = tokens[num_consumed:] # remove qualifier list tokens
        base.check_if_all_tokens_are_blank(tokens)
        if len(rhs) != 1:
            raise error.SyntaxError("expected 1 identifier with optional parameter list")
        rhs_tokens = base.tokenize(rhs[0])
        name = rhs_tokens.pop(0)
        if not name.isidentifier():
            raise error.SyntaxError("expected identifier")
        params = []
        if len(rhs_tokens) and rhs_tokens.pop(0)=="(":
            params,num_consumed2 = base.get_top_level_operands(rhs_tokens)  
            rhs_tokens = rhs_tokens[num_consumed2:]
            if not rhs_tokens.pop(0) == ")":
                raise error.SyntaxError("expected ')'")
            if not len(params):
                raise error.SyntaxError("expected at least 1 parameter")
        return name, attributes, params
    
    def parse_fortran_argument(self,statement):
        tokens = base.tokenize(statement)
        parts,num_consumed = base.get_top_level_operands(tokens,separators="=")
        if len(parts) == 2:
            return (parts[0],parts[1])
        elif len(parts) == 1:
            return (None,parts[0])
        else:
            raise error.SyntaxError("expected single expression or named argument expression")
    
    def map_fortran_args_to_positional_args(self,fortran_args,
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
            args1,_ = base.get_top_level_operands(base.tokenize(fortran_args))
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
            fortran_arg_list = [self.parse_fortran_argument(expr) for expr in args1]
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
    
    def _parse_f77_character_type(self,statement,parse_all=False):
        tokens = base.tokenize(statement,padded_size=3)
        char_type = tokens[0]
        char_kind = None 
        char_len  = "1"
        if base.compare_ignore_case(tokens[0:3],["character","*","("]):
            total_num_consumed = 3
            args, num_consumed = base.get_top_level_operands(tokens[3:])
            total_num_consumed += num_consumed
            if tokens[total_num_consumed:total_num_consumed+1] != [")"]:
                raise error.SyntaxError("missing ')'")
            total_num_consumed += 1
            base.check_if_all_tokens_are_blank(tokens[total_num_consumed:],parse_all)
            if len(args) == 1:
                char_len = self.map_fortran_args_to_positional_args(
                  args,
                  positional_args=["len"],
                  defaults=["1"],allow_named_args=False
                )[0]
                return (char_type, char_len, None, [], total_num_consumed)
            else:
                raise error.SyntaxError("expected a single unnamed 'len' expression")        
        elif base.compare_ignore_case(tokens[0:2],["character","*"]):
            char_len = tokens[2]
            if not char_len.isidentifier():
                try:
                   int(char_len) 
                except ValueError:
                    raise error.SyntaxError("if no parantheses are used, character len argument must be identifier or integer")
            base.check_if_all_tokens_are_blank(tokens[3:],parse_all)
            return (char_type, char_len, None, [], 3)
        elif base.compare_ignore_case(tokens[0],"character"):
            base.check_if_all_tokens_are_blank(tokens[1:],parse_all)
            return (char_type, char_len, None, [], 1)
        else:
            raise error.SyntaxError("expected 'character'") 
                
    def parse_fortran_character_type(self,statement,parse_all=False):
        tokens = base.tokenize(statement,padded_size=2)
        if base.compare_ignore_case(tokens[0:2],["character","("]):
            total_num_consumed = 2 
            args, num_consumed = base.get_top_level_operands(tokens[total_num_consumed:])
            total_num_consumed += num_consumed
            if tokens[total_num_consumed:total_num_consumed+1] != [")"]:
                raise error.SyntaxError("missing ')'")
            total_num_consumed += 1
            base.check_if_all_tokens_are_blank(tokens[total_num_consumed:],parse_all)
            if len(args) in [1,2]:
                values = self.map_fortran_args_to_positional_args(args,["len","kind"],defaults=["1",None])
                char_len, char_kind = values
                return (tokens[0], char_len, char_kind, [], total_num_consumed)
            else:
                raise error.SyntaxError("expected 1 or 2 comma-separated character length and kind expressions")
        else:
            return self._parse_f77_character_type(tokens,parse_all)
    
    def _parse_f77_basic_type(self,statement,parse_all=False):
        tokens = base.tokenize(statement,padded_size=3)
        if base.compare_ignore_case(tokens[0:2],["double","precision"]):
            base.check_if_all_tokens_are_blank(tokens[2:],parse_all)
            return ("real", None, "c_double", [], 2)
        elif base.compare_ignore_case(tokens[0:2],["double","complex"]):
            base.check_if_all_tokens_are_blank(tokens[2:],parse_all)
            return ("complex", None, "c_double_complex", [], 2)
        elif tokens[0].lower() in _basic_types:
            basic_type = tokens[0]
            if base.compare_ignore_case(tokens[1:3],["*","("]):
                total_num_consumed = 3
                args, num_consumed = base.get_top_level_operands(tokens[3:])
                total_num_consumed += num_consumed
                if tokens[total_num_consumed:total_num_consumed+1] != [")"]:
                    raise error.SyntaxError("missing ')'")
                total_num_consumed += 1
                base.check_if_all_tokens_are_blank(tokens[total_num_consumed:],parse_all)
                if len(args) == 1:
                    basic_kind = self.map_fortran_args_to_positional_args(args,
                                                                          positional_args=["kind"],
                                                                          defaults=[None],allow_named_args=False)[0]
                    return (basic_type, None, basic_kind, [], total_num_consumed)
                else:
                    raise error.SyntaxError("expected a single unnamed 'kind' expression")        
            elif base.compare_ignore_case(tokens[1:2],["*"]):
                basic_kind = tokens[2]
                if not basic_kind.isidentifier():
                    try:
                       int(basic_kind) 
                    except ValueError:
                        raise error.SyntaxError("if no parantheses are used, kind argument must be identifier or integer")
                base.check_if_all_tokens_are_blank(tokens[3:],parse_all)
                return (basic_type, None, basic_kind, [], 3)
            else:
                base.check_if_all_tokens_are_blank(tokens[1:],parse_all)
                return (basic_type, None, None, [], 1)
        else:
                raise error.SyntaxError("expected 'character'") 
    
    def parse_fortran_basic_type(self,statement,parse_all=False):
        """:note: character is handleded in a dedicated function."""
        tokens = base.tokenize(statement,padded_size=2)
        if (tokens[0].lower() in _basic_types
           and base.compare_ignore_case(tokens[1:2],["("])):
            total_num_consumed = 2 
            args, num_consumed = base.get_top_level_operands(tokens[total_num_consumed:])
            total_num_consumed += num_consumed
            if tokens[total_num_consumed:total_num_consumed+1] != [")"]:
                raise error.SyntaxError("missing ')'")
            total_num_consumed += 1
            base.check_if_all_tokens_are_blank(tokens[total_num_consumed:],parse_all)
            if len(args) == 1:
                basic_kind = self.map_fortran_args_to_positional_args(args,["kind"],defaults=[None])[0]
                return (tokens[0],None,basic_kind,[],total_num_consumed)
            else:
                raise error.SyntaxError("expected a single kind expressions")
        else:
            return self._parse_f77_basic_type(tokens,parse_all)
    
    def parse_fortran_derived_type(self,statement,type_or_class="type",parse_all=True):
        tokens = base.tokenize(statement,padded_size=2)
        if base.compare_ignore_case(tokens[0:2],[type_or_class,"("]):
            total_num_consumed = 2
            types,num_consumed = base.get_top_level_operands(tokens[total_num_consumed:],join_operand_tokens=False)
            total_num_consumed += num_consumed
            if tokens[total_num_consumed:total_num_consumed+1] != [")"]:
                raise error.SyntaxError("missing ')'")
            total_num_consumed += 1
            base.check_if_all_tokens_are_blank(tokens[total_num_consumed:],parse_all)
            if len(types) == 1:
                first_entry = types[0]
                if (
                    len(first_entry) == 1 
                    and ( first_entry[0].isidentifier() 
                          or first_entry[0] == "*" )
                ):
                    return (tokens[0],None,first_entry[0],[],total_num_consumed)
                elif (  # parametrized types
                    len(first_entry) >= 2 
                    and first_entry[0].isidentifier() 
                    and first_entry[1] == "("
                ):
                    params,num_params_tokens = base.get_top_level_operands(first_entry[2:])
                    if first_entry[2+num_params_tokens:] != [")"]:
                        raise error.SyntaxError("missing ')'")
                    base.check_if_all_tokens_are_blank(first_entry[2+num_params_tokens+1:],parse_all)
                    return (tokens[0],None,first_entry[0],params,total_num_consumed)
                else:
                    raise error.SyntaxError("expected identifier, '*', or identifier plus parameter list")
            else:
                raise error.SyntaxError("expected single identifier, '*', or identifier plus parameter list")
        else:
            raise error.SyntaxError("expected 'type ('")
    
    def parse_fortran_datatype(self,statement,parse_all=False):
        tokens = base.tokenize(statement,padded_size=1)
        if base.compare_ignore_case(tokens[0],"type"):
            return self.parse_fortran_derived_type(tokens,"type",parse_all)
        elif base.compare_ignore_case(tokens[0],"class"):
            return self.parse_fortran_derived_type(tokens,"class",parse_all)
        elif base.compare_ignore_case(tokens[0],"character"):
            return self.parse_fortran_character_type(tokens,parse_all)
        else:
            return self.parse_fortran_basic_type(tokens,parse_all)
            
    
    def parse_declaration(self,statement):
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
        tokens = base.tokenize(statement,10)
        is_character_declaration = tokens[0].lower() == "character"
        is_f77_character_declaration = base.compare_ignore_case(tokens[0:2],["character","*"])
    
        # handle datatype
        datatype, length, kind, params, num_consumed = self.parse_fortran_datatype(tokens)
        
        datatype_raw = "".join(tokens[:num_consumed])
       
        # parse raw qualifiers and variables
        tokens = tokens[num_consumed:] # remove type part tokens
        try:
            qualifiers_raw, variables_raw, num_consumed = self._parse_raw_qualifiers_and_rhs(tokens)
            tokens = tokens[num_consumed:] # remove qualifier list tokens
        except error.SyntaxError: # superfluous comma allowed between character*<expr> and first identifier
            if is_f77_character_declaration and tokens[0] == ",":
                tokens = tokens[1:]
                qualifiers_raw, variables_raw, num_consumed = self._parse_raw_qualifiers_and_rhs(tokens)
                tokens = tokens[num_consumed:] # remove qualifier list tokens
            else:
                raise
        base.check_if_all_tokens_are_blank(tokens)
        # analyze qualifiers
        qualifiers = []
        dimension_bounds = []
        for qualifier in qualifiers_raw:
            qualifier_tokens = base.tokenize(qualifier)
            if qualifier_tokens[0].lower() == "dimension":
                # ex: dimension ( 1, 2, 1:n )
                qualifier_tokens = base.tokenize(qualifier)
                if (len(qualifier_tokens) < 4
                   or not base.compare_ignore_case(qualifier_tokens[0:2],["dimension","("])
                   or qualifier_tokens[-1] != ")"):
                    raise error.SyntaxError("could not parse 'dimension' qualifier")
                dimension_bounds,_  = base.get_top_level_operands(qualifier_tokens[2:-1])
            elif qualifier_tokens[0].lower() == "intent":
                # ex: intent ( inout )
                qualifier_tokens2 = base.tokenize(base.remove_whitespace(qualifier))
                if (len(qualifier_tokens2) != 4 
                   or not base.compare_ignore_case(qualifier_tokens2[0:2],["intent","("])
                   or qualifier_tokens2[-1] != ")"
                   or qualifier_tokens2[2].lower() not in ["out","inout","in"]):
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
            assignment_operands,_ = base.get_top_level_operands(base.tokenize(var),
                                                           separators=[separator])
            if len(assignment_operands) == 1:
                var_lhs = assignment_operands[0]
                var_rhs = None
            elif len(assignment_operands) == 2:
                var_lhs,var_rhs = assignment_operands
            else:
                raise error.SyntaxError("expected single variable expression or (pointer) assigment")
            if is_character_declaration:
                length_operands,_ = base.get_top_level_operands(base.tokenize(var_lhs),
                                                           separators=["*"])
                if len(length_operands) == 2:
                    var_lhs, length = length_operands 
            var_lhs_tokens = base.tokenize(var_lhs)
            var_name = var_lhs_tokens[0]
            if not var_name.isidentifier():
                raise error.SyntaxError("expected identifier")
            if (len(var_lhs_tokens) > 2
               and var_lhs_tokens[1] == "("):
                var_bounds, num_consumed = base.get_top_level_operands(var_lhs_tokens[2:])
                if not len(var_bounds):
                    raise error.SyntaxError("expected at least one argument for variable '{}'".format(var_name))
                if var_lhs_tokens[2+num_consumed:] != [")"]:
                    raise error.SyntaxError("missing ')'")
            elif len(var_lhs_tokens) > 1:
                raise error.SyntaxError("unexpected tokens after '{}': '{}'".format(var_name,"','".join(var_lhs_tokens[1:])))
            else:
                var_bounds = []
            variables.append((var_name,var_bounds,var_rhs))
        return (datatype, length, kind, params, qualifiers, dimension_bounds, variables, datatype_raw, qualifiers_raw) 
    
    def __parse_array_spec(self,tokens):
        array_specs = []
        array_spec_exprs, num_consumed = base.get_top_level_operands(tokens,
                                                                join_operand_tokens=False)
        tokens.pop_front(num_consumed)
        if not len(array_spec_exprs):
            raise error.SyntaxError("expected at least one array specification")    
        array_spec_tokens = ts.TokenStream([])
        for array_spec_tokens1 in array_spec_exprs:
            array_spec_tokens.setup(array_spec_tokens1)
            var_name = array_spec_tokens.pop_front()
            if not var_name.isidentifier():
                raise error.SyntaxError("expected identifier")
            if not array_spec_tokens.pop_front_equals("("):
                raise error.SyntaxError("expected '('")
            var_bounds, num_consumed = base.get_top_level_operands(array_spec_tokens)
            array_spec_tokens.pop_front(num_consumed)
            if not len(var_bounds):
                raise error.SyntaxError("expected at least one argument for variable '{}'".format(var_name))
            if not array_spec_tokens.pop_front_equals(")"):
                raise error.SyntaxError("missing ')'")
            array_spec_tokens.check_if_remaining_tokens_are_blank()
            array_specs.append((var_name,var_bounds))
        tokens.check_if_remaining_tokens_are_blank()
        return array_specs
    
    def parse_dimension_statement(self,statement):
        """:return: Array specifications: List of tuples containing variable name and a
                    list of lower and upper bounds per dimension (in that order).
        :param statement: A string expression or a list of tokens.
        """
        tokens = ts.TokenStream(base.tokenize(statement,padded_size=5))
        if tokens.pop_front().lower() not in ["dimension","dim"]:
            raise error.SyntaxError("expected 'dimension' or 'dim'")
        return self.__parse_array_spec(tokens)
    
    def parse_external_statement(self,statement):
        """:return: List of identifiers of procedures.
        :param statement: A string expression or a list of tokens."""
        tokens = ts.TokenStream(base.tokenize(statement,padded_size=3))
        if tokens.pop_front().lower() not in ["external"]:
            raise error.SyntaxError("expected 'external'")
        next_token = tokens[0]
        if next_token == "::":
            tokens.pop_front()
        elif not next_token.isidentifier():
            raise error.SyntaxError("expected '::' or identifier")
            
        procedure_names, num_consumed = base.get_top_level_operands(tokens)
        if len([p for p in procedure_names if not p.isidentifier()]):
            raise error.SyntaxError("expected list of identifiers")
        tokens.pop_front(num_consumed)
        tokens.check_if_remaining_tokens_are_blank()
        return procedure_names
    
    def parse_derived_type_statement(self,statement):
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
        tokens = base.tokenize(statement,padded_size=10)
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
            attributes,_  = base.get_top_level_operands(tokens)
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
            parameters,_  = base.get_top_level_operands(tokens[:-1])
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
    
    def parse_public_or_private_statement(self,statement,kind):
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
        tokens      = base.tokenize(statement,padded_size=1)
        kind_expr   = tokens.pop(0)
        identifiers = []
        operators   = []
        assignments = []
        if kind_expr.lower() != kind:
            raise error.SyntaxError("expected '{}'".format(kind))
        if len(tokens) > 1 and tokens[0] == "::":
            tokens.pop(0)
        if len(tokens) and tokens[0].isidentifier():
            operands, num_consumed = base.get_top_level_operands(tokens,join_operand_tokens=False)
            for expr in operands:
                if len(expr) == 1 and expr[0].isidentifier():
                    identifiers.append(expr[0])
                elif (len(expr) == 4
                   and base.compare_ignore_case(expr[0:2],["operator","("])
                   and expr[3] == ")"):
                    # todo: check if operator is valid
                    identifiers.append("".join(expr))
                elif (len(expr) == 4
                   and base.compare_ignore_case(expr[0:2],["assignment","("])
                   and expr[3] == ")"):
                    # todo: check if assignment is valid
                    identifiers.append("".join(expr))
                else:
                    raise error.SyntaxError("expected identifier or 'operator' + '(' + operator expression + ')'")
            tokens = tokens[num_consumed:] 
        base.check_if_all_tokens_are_blank(tokens)
        return kind_expr, identifiers
    
    def parse_public_statement(self,statement):
        """:see: parse_public_or_private_statement"""
        return self.parse_public_or_private_statement(statement,"public")
    
    def parse_private_statement(self,statement):
        """:see: parse_public_or_private_statement"""
        return self.parse_public_or_private_statement(statement,"private")
        
    def parse_implicit_statement(self,statement):
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
        tokens = base.tokenize(statement,padded_size=2)    
        implicit_specs = []
        if tokens.pop(0).lower() != "implicit":
            raise error.SyntaxError("expected 'implicit'")
        rules,num_consumed = base.get_top_level_operands(tokens)
        tokens = tokens[num_consumed:]
        base.check_if_all_tokens_are_blank(tokens)
        if len(rules)==1 and rules[0].lower() == "none":
            implicit_specs.append((None,None,None,[]))
            base.check_if_all_tokens_are_blank(rules[1:])
        elif len(rules):
            taken_letters = set()
            # example: implicit character (c), integer(mykind) (j,k,m,n)       
            for rule in rules:
                rule_tokens  = base.tokenize(rule)
                datatype_tokens = base.next_tokens_till_open_bracket_is_closed(
                        rule_tokens,open_brackets=1,brackets=(None,")"))
                if len(datatype_tokens) == len(rule_tokens):
                    # In this case, last brackets do not belong to datatype_tokens
                    # example: implicit character (c)           
                    # example: implicit double precision (a-z)           
                    # example: implicit double complex (c-d)           
                    if len(rule_tokens) > 1 and rule_tokens[1].isidentifier():
                        # example: implicit double precision (a-z)           
                        last_datatype_token = 1 
                    else:
                        # example: implicit character (c)
                        last_datatype_token = 0
                    datatype_tokens = rule_tokens[:last_datatype_token+1]
                    first_letter_tokens = rule_tokens[last_datatype_token+1:]
                else:
                    # example: implicit integer(mykind) (j,k,m,n)       
                    first_letter_tokens = rule_tokens[len(datatype_tokens):]
                f_type, f_len, f_kind, params, num_consumed = self.parse_fortran_datatype(datatype_tokens)
                assigned_first_letters = []
                base.check_if_all_tokens_are_blank(datatype_tokens[num_consumed:])
                #rule_tokens = rule_tokens[num_consumed:]
                #letters = []
                if (len(first_letter_tokens) and first_letter_tokens[0] == "("
                   and first_letter_tokens[-1] == ")"):
                    letter_range_expressions,num_consumed = base.get_top_level_operands(first_letter_tokens[1:-1])
                    base.check_if_all_tokens_are_blank(first_letter_tokens[num_consumed+2:])
                    
                    for expr in letter_range_expressions:
                        letter_range = base.expand_letter_range(expr)
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
    
    def parse_directive(self,directive,tokens_to_omit_at_begin=0):
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
        tokens = base.tokenize(directive)
        it = 0
        result = []
        while len(tokens):
            tk = tokens.pop(0) 
            if tk == "(":
                rest = base.next_tokens_till_open_bracket_is_closed(tokens,1)
                result[-1] = "".join([result[-1],tk]+rest)
                tokens = tokens[len(rest):]
            elif tk.isidentifier() or tk.endswith("$"):
                result.append(tk)
            elif tk != ",": # optional commas are allowed between clauses
                raise error.SyntaxError("unexpected clause or identifier token: {}".format(tk))
        return result[tokens_to_omit_at_begin:]
    
    def parse_acc_directive(self,statement):
        """Parses an OpenACC directive, returns a 4-tuple of its sentinel, its kind (as list of tokens),
        its arguments and its clauses (as list of tokens). 
    
        :Example:
    
        Given
          `!$acc enter data copyin(a,b) async`,
        this routine returns
          ('!$',['acc','enter','data'],['copyin(a,b)','async'])
        """
        directive_parts = self.parse_directive(statement)
        directive_kinds = [ # order is important
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
                if base.compare_ignore_case(directive_parts[0:length-1]+last_directive_token_parts[0:1],kind):
                    directive_kind = kind
                    if len(last_directive_token_parts) > 1:
                        _, directive_args = self.parse_acc_clauses([directive_parts[length-1]])[0] 
                    clauses = directive_parts[length:]
                    break
        if not len(directive_kind):
            raise error.SyntaxError("could not identify kind of directive")
        return sentinel, directive_kind, directive_args, clauses
    
    def parse_acc_clauses(self,clauses):
        """Converts OpenACC-like clauses into tuples with the name
        of the clause as first argument and the list of arguments of the clause
        as second argument, might be empty.
        :param list clauses: A list of acc clause expressions (strings).
        """
        result = []
        for expr in clauses:
            tokens = base.tokenize(expr)
            if len(tokens) == 1 and tokens[0].isidentifier():
                result.append((tokens[0],[]))
            elif (len(tokens) > 3 
                 and tokens[0].isidentifier()
                 and tokens[1] == "(" 
                 and tokens[-1] == ")"):
                args1,_  = base.get_top_level_operands(tokens[2:-1], [","],[]) # clip the ')' at the end
                args = []
                current_group = None
                for arg in args1:
                    arg_parts = base.tokenize(arg)
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
    
    def check_acc_clauses(self,directive_kind,clauses):
        """Check if the used acc clauses are legal for the given directive.
        :param list directive_kind: List of directive tokens as produced by parse_acc_directive.
        :param list clauses: list of tuples as produced by parse_acc_clauses.
        """
        key = " ".join(directive_kind[1:]).lower()
        if key in acc_clauses.acc_clauses:
            for name,_ in clauses:
                if name.lower() not in acc_clauses.acc_clauses[key]:
                    raise error.SyntaxError("clause '{}' not allowed on 'acc {}' directive".format(name.lower(),key))
        elif len(clauses):
            raise error.SyntaxError("No clauses allowed on 'acc {}' directive".format(key))
    
    def parse_cuf_kernel_call(self,statement):
        """Parses a CUDA Fortran kernel call statement.
        Decomposes it into kernel name, launch parameters,
        and kernel arguments.
        
        :Example:
    
        `call mykernel<<grid,dim>>(arg0,arg1,arg2(1:n))`
    
        will result in a triple
    
        ("mykernel",["grid","dim"],["arg0","arg1","arg2(1:n)"])
        """
        tokens = base.tokenize(statement)
        if tokens.pop(0).lower() != "call":
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'call'")
        kernel_name = tokens.pop(0)
        # parameters
        if not kernel_name.isidentifier():
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected 'identifier'")
        if tokens.pop(0) != "<<<":
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '<<<'")
        params_tokens = base.next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, brackets=("<<<",">>>"), keepend=False)
        for i in range(0,len(params_tokens)):
            tokens.pop(0)
        if tokens.pop(0) != ">>>":
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '>>>'")
        # arguments
        if tokens.pop(0) != "(":
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected '('")
        args_tokens = base.next_tokens_till_open_bracket_is_closed(tokens, open_brackets=1, keepend=False)
        for i in range(0,len(args_tokens)):
            tokens.pop(0)
        if tokens.pop(0) != ")":
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: expected ')'")
        params,_  = base.get_top_level_operands(params_tokens,terminators=[])
        args,_  = base.get_top_level_operands(args_tokens,terminators=[])
        if len(tokens):
            raise error.SyntaxError("could not parse CUDA Fortran kernel call: trailing text")
        return kernel_name, params, args
    
    def _parse_named_label(self,tokens):
        """param ts.TokenStream tokens: A token stream."""
        if len(tokens) > 1:
            first: str = tokens[0]
            if ( first.isidentifier()
                 and tokens[1] == ":" ):
                tokens.pop_front(2)
                return first
        raise error.SyntaxError("expected identifier + ':'")
    

    def parse_block_statement(self,statement):
        tokens = ts.TokenStream(statement)
        try:
            named_label = self._parse_named_label(tokens)
        except error.SyntaxError:
            named_label = None
        #
        if not tokens.pop_front_equals_ignore_case("block"):
            raise error.SyntaxError("expected 'block'")
        tokens.check_if_remaining_tokens_are_blank()
        return named_label
    
    def _parse_single_argument(self,stream):
        """Parses expressions of the form <tokens>+')'.
        :see: base.next_tokens_till_open_bracket_is_closed
        """
        cond_tokens = base.next_tokens_till_open_bracket_is_closed(
          stream,
          open_brackets = 1,
          keepend = False
        )
        if not len(cond_tokens):
            raise error.SyntaxError("expected argument")
        stream.pop_front(len(cond_tokens))
        if not stream.pop_front_equals(")"):
            raise error.SyntaxError("missing ')'")
        return " ".join(cond_tokens)
    
    def parse_if_statement(self,statement):
        tokens = ts.TokenStream(statement)
        try:
            named_label = self._parse_named_label(tokens)
        except error.SyntaxError:
            named_label = None
        #
        if tokens.pop_front_equals_ignore_case("if","("):
            cond = self._parse_single_argument(tokens)
            if not tokens.pop_front_equals_ignore_case("then"):
                raise error.SyntaxError("expected 'then'")
        else:
            raise error.SyntaxError("expected 'if' + '('")
        tokens.check_if_remaining_tokens_are_blank()
        return (named_label, cond)
    
    def parse_else_if_statement(self,statement):
        tokens = ts.TokenStream(statement)
        self._parse_combined_keyword(tokens,"else","if")
        if tokens.pop_front_equals_ignore_case("("):
            cond = self._parse_single_argument(tokens)
            if not tokens.pop_front_equals_ignore_case("then"):
                raise error.SyntaxError("expected 'then'")
        else:
            raise error.SyntaxError("expected '('")
        tokens.check_if_remaining_tokens_are_blank()
        return cond
    
    def parse_else_statement(self,statement):
        tokens = ts.TokenStream(statement)
        if not tokens.pop_front_equals_ignore_case("else"):
            raise error.SyntaxError("expected 'else'")
        tokens.check_if_remaining_tokens_are_blank()
        return None
    
    def parse_where_statement(self,statement):
        """:returns: A named label or None and a mask expression."""
        tokens = ts.TokenStream(statement)
        try:
            named_label = self._parse_named_label(tokens)
        except error.SyntaxError:
            named_label = None
        if tokens.pop_front_equals_ignore_case("where","("):
            mask = self._parse_single_argument(tokens)
        else:
            raise error.SyntaxError("expected 'where' + '('")
        tokens.check_if_remaining_tokens_are_blank()
        return named_label, mask
    
    def parse_else_where_statement(self,statement):
        """:returns: A mask expr or None."""
        tokens = ts.TokenStream(statement)
        self._parse_combined_keyword(tokens,"else","where")
        if len(tokens) and tokens[0] == "(":
            tokens.pop_front() 
            mask = self._parse_single_argument(tokens)
        tokens.check_if_remaining_tokens_are_blank()
        return mask
    
    def parse_select_case_statement(self,statement):
        """:returns: A named label and the argument.
        """
        tokens = ts.TokenStream(statement)
        try:
            named_label = self._parse_named_label(tokens)
        except error.SyntaxError:
            named_label = None
        #
        argument = None
        self._parse_combined_keyword(tokens,"select","case")
        if tokens.pop_front_equals_ignore_case("("):
            argument = self._parse_single_argument(tokens)
        else:
            raise error.SyntaxError("expected '('")
        return (named_label, argument)
    
    def parse_case_statement(self,statement):
        tokens = ts.TokenStream(statement)
        if tokens.pop_front_equals_ignore_case("case","("):
            values,num_consumed = base.get_top_level_operands(
              tokens,terminators=[")"])
            tokens.pop_front(num_consumed)
            if not tokens.pop_front_equals_ignore_case(")"):
                raise error.SyntaxError("expected ')'")
        else:
            raise error.SyntaxError("expected 'case' + '('")
        tokens.check_if_remaining_tokens_are_blank()
        return values

    def parse_case_default_statement(self,statement):
        tokens = ts.TokenStream(statement)
        if not tokens.pop_front_equals_ignore_case("case","default"):
            self._raise_unclassified_error(tokens,"expected 'case' + 'default'")
    
    def parse_end_statement(self,statement,kind):
        """Parses statements in the form:
        END
        END kind [label]
        ENDkind [label]
        :return: label or None.
        """
        tokens = ts.TokenStream(statement)
        if kind == None:
            if not tokens.pop_front_equals_ignore_case("end"):
                raise error.SyntaxError("expected 'end'")
            named_label = None
        else:
            self._parse_combined_keyword(tokens,"end",kind)
            if tokens.empty():
                named_label = None
            else:
                named_label = tokens.pop_front()
                if not named_label.isidentifier():
                    raise error.SyntaxError("expected identifier")
        tokens.check_if_remaining_tokens_are_blank()
        return named_label


    def _parse_combined_keyword(self,tokens,first,second):
        if base.compare_ignore_case(tokens[0],first+second):
            tokens.pop_front()
        elif not tokens.pop_front_equals_ignore_case(first,second):
            raise self._raise_unclassified_error(tokens,"expected '{}' + '{}'".format(
              first,second
            ))

    def parse_unconditional_goto_statement(self,statement):
        """Parses unconditional GO TO statements:
        GO TO numeric-label|loop-labe|ASSIGN-label
        """
        tokens = ts.TokenStream(statement)
        self._parse_combined_keyword(tokens,"go","to")
        unconditional_goto_label = tokens.pop_front()
        if (not unconditional_goto_label.isidentifier()
            and not unconditional_goto_label.isnumeric()):
             self._raise_classified_error(tokens,"expected identifier")
        tokens.check_if_remaining_tokens_are_blank()
        return unconditional_goto_label

    def parse_assigned_goto_statement(self,statement):
        """Parses assigned GO TO statements:
        GO TO <ASSIGN-label> [[,](numeric-label1[, numeric-label2])]"""
        tokens = ts.TokenStream(statement)
        self._parse_combined_keyword(tokens,"go","to")
        assigned_goto_label = tokens.pop_front()
        if not assigned_goto_label.isidentifier():
            raise self._raise_classified_error(tokens,"'{}' is no identifier".format(
              assigned_goto_label)
            )
        if tokens[0] == ",":
            tokens.pop_front()
            if not tokens.pop_front_equals_ignore_case("("):
                raise self._raise_classified_error(tokens,"expected '('")
        elif not tokens.pop_front_equals("("):
            raise self._raise_classified_error(tokens,"expected '(' or ','")
        assigned_goto_numeric_labels, num_consumed =  base.get_top_level_operands(tokens)
        for label in assigned_goto_numeric_labels:
              if not label.isnumeric():
                  raise self._raise_classified_error(tokens,"'{}' is no numeric statement label".format(label))
        tokens.pop_front(num_consumed)
        if not tokens.pop_front_equals( ")"):
            raise self._raise_classified_error(tokens,"expected ')'")
        tokens.check_if_remaining_tokens_are_blank()
        return (assigned_goto_label, assigned_goto_numeric_labels)

    def parse_computed_goto_statement(self,statement):
        """Parses computed GO TO statements:
        GO TO (numeric-label1[, numeric-label2, ...]) [,] <integer or real expr>
        """
        tokens = ts.TokenStream(statement)
        self._parse_combined_keyword(tokens,"go","to")
        if not tokens.pop_front_equals("("):
            self._raise_unclassified_error(tokens,"expected '('")
        computed_goto_numeric_labels, num_consumed = base.get_top_level_operands(tokens)
        if len(computed_goto_numeric_labels):
            for label in computed_goto_numeric_labels:
                  if not label.isnumeric():
                      raise self._raise_classified_error(tokens,"'{}' is no numeric statement label".format(label))
            tokens.pop_front(num_consumed)
        else:
            raise self._raise_classified_error(tokens,"expected at least one numeric label")
        if not tokens.pop_front_equals( ")"):
            raise self._raise_classified_error(tokens,"expected ')'")
        if not len(tokens):
            raise self._raise_classified_error(tokens,"expected ',' and/or integer/real expression")
        if tokens[0] == ",":
            tokens.pop_front()
        computed_goto_expr = "".join(tokens.pop_front(len(tokens)))
        if not len(computed_goto_expr):
            raise self._raise_classified_error(tokens,"expected integer/real expression")
        return (computed_goto_numeric_labels, computed_goto_expr)
    

    def _parse_cycle_or_exit_statement(self,statement,kind):
        """Parses expression of the form:
        {kind} [named-do-label]
        :return: Identifier or None.
        """
        tokens = ts.TokenStream(statement)
        if not tokens.pop_front_equals_ignore_case(kind):
            raise self._raise_classified_error(tokens,"expected '{}'".format(kind))
        if len(tokens):
            named_do_label = tokens.pop_front()
            if not named_do_label.isidentifier():
                raise self._raise_classified_error(tokens,"expected identifier")
            return named_do_label
        else:
            return None

    def parse_cycle_statement(self,statement):
        """Parses expression of the form:
        CYCLE [named-do-label]
        :return: Identifier or None.
        """
        return self._parse_cycle_or_exit_statement(statement,"cycle") 

    def parse_exit_statement(self,statement):
        """Parses expression of the form:
        EXIT [named-do-label]
        :return: Identifier or None.
        """
        return self._parse_cycle_or_exit_statement(statement,"exit") 

    def parse_return_statement(self,statement):
        """Parse return statement:
        RETURN [alternate_return_ordinal_number_expression]
        :return: Expression as string or None.
        """
        if not tokens.pop_front_equals_ignore_case("return"):
            raise error.SyntaxError("expected 'return'")
        if len(tokens):
            return tokens.pop_front()
        else:
            return None

    def parse_do_statement(self,statement):
        """ High-level parser for 
        `[named_label :] DO [numeric_label] [( WHILE ( cond ) | var = lbound, ubound [,stride] )]`
        where `named_label`, and `var` are identifiers, `numeric_label` is an integer string,
        and `cond`, `lbound`, `ubound`, and `stride` are arithmetic expressions.
    
        :return: Tuple consisting of `named_label`, `numeric_label`, `cond`, 
                 `var`, `lbound`, `ubound`, `stride` (in that order).
                 Optional parts are returned as None if they could not be found.
                 Named and numeric label cannot be specified together.
    
        :Examples:
    
        `mylabel: do i = 1,N+m`,
        `do`,
        `do 50`,
        `do 50 while ( true )`,
    
        result in
    
        ('mylabel', None, None, 'i', '1', 'N+m', None),
        (None, None, None, None, None, None, None),
        (None, 50, None, None, None, None, None),
        (None, 50, 'true', None, None, None, None)
        """
        tokens = ts.TokenStream(statement)
        try:
            named_label = self._parse_named_label(tokens)
        except error.SyntaxError:
            named_label = None
        #
        if not tokens.pop_front_equals_ignore_case("do"):
            raise error.SyntaxError("expected 'do'")
        numeric_label = None
        var = None
        cond = None
        lbound = None
        ubound = None
        stride = None
        if not tokens.empty():
            tk = tokens[0] 
            if tk.isnumeric():
                numeric_label = tk 
                if named_label != None:
                    raise error.SyntaxError("named label and numeric label cannot be specified together")
                tokens.pop_front()
        if tokens.size() >= 2:
            next_two_tokens = tokens.pop_front(2) 
            if base.compare_ignore_case(next_two_tokens,["while","("]):
                cond = self._parse_single_argument(tokens)
            elif ( next_two_tokens[0].isidentifier()
                   and next_two_tokens[1] == "=" ):
                var = next_two_tokens[0]
                range_vals,consumed_tokens  = base.get_top_level_operands(tokens) 
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
                tokens.pop_front(consumed_tokens)
            else:
                raise error.SyntaxError("expected 'while' + '(' or identifier + '=' ")
        tokens.check_if_remaining_tokens_are_blank()
        return (named_label, numeric_label, cond, var, lbound, ubound, stride)
    
    def parse_deallocate_statement(self,statement):
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
        tokens = base.tokenize(statement)
        if not tokens.pop(0).lower() == "deallocate":
            raise error.SyntaxError("expected 'deallocate'")
        if tokens.pop(0) != "(":
            raise error.SyntaxError("expected '('")
        args,_  = base.get_top_level_operands(tokens) # : [...,"b(-1:m,n)"]
        for arg in args:
            parts,_ = base.get_top_level_operands(base.tokenize(arg),separators=["%"])
            child_tokens = base.tokenize(parts[-1])
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
    
    def parse_allocate_statement(self,statement):
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
        tokens = base.tokenize(statement)
        if not tokens.pop(0).lower() == "allocate":
            raise error.SyntaxError("expected 'allocate'")
        if tokens.pop(0) != "(":
            raise error.SyntaxError("expected '('")
        args,_  = base.get_top_level_operands(tokens) # : [...,"b(-1:m,n)"]
        for arg in args:
            parts,_ = base.get_top_level_operands(base.tokenize(arg),separators=["%"])
            child_tokens = base.tokenize(parts[-1])
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
                if base.all_tokens_are_blank(child_tokens):
                    var_tuple = (varname, [])
                    result1.append(var_tuple)
                elif child_tokens[0] == "(":
                    child_tokens.pop(0)
                    ranges,_  = base.get_top_level_operands(child_tokens) # : ["-1:m", "n"]
                    var_tuple = (varname, [])
                    for r in ranges:
                        range_tokens,_  = base.get_top_level_operands(base.tokenize(r),
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
    
    #def parse_arith_expr(self,statement,logical_ops=False,max_recursions=0):
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
    #        tokens = base.tokenize(statement)
    #        open_bracket = tokens[0] == "(":
    #        top_level_operands,_ = base.get_top_level_operands(tokens,separators=separators,keep_separators=True)
    #        if len(top_level_operands) == 1 and open_bracket:
    #             if tokens[-1] == ")":
    #                 raise error.SyntaxError("missing ')'")
    #             result = []
    #             # todo: could be complex value
    #             result.append(inner_(tokens[1:-1]))
    #             return
    #        elif len(top_level_operands) == 1:
    #            return top_level_operands
    #        elif depth < max_recursions:
    #            inner_(
    
    def parse_assignment(self,statement,parse_rhs=False):
        tokens = base.tokenize(statement)
        parts,_ = base.get_top_level_operands(tokens,separators=["="])
        if len(parts) != 2:
            raise error.SyntaxError("expected left-hand side and right-hand side separated by '='")
        return (parts[0], parts[1])
    
    def parse_lvalue(self,statement):
        tokens = base.tokenize(statement)
        parts,_ = base.get_top_level_operands(tokens,separators=["%"],
                                         join_operand_tokens=False)
        result = []
        for part_tokens in parts:
            ident = part_tokens.pop(0)
            if not ident.isidentifier():
                raise error.SyntaxError("expected identifier")
            if len(part_tokens) and part_tokens[0] == "(":
                part_tokens.pop(0)
                args,num_argument_tokens = base.get_top_level_operands(part_tokens)
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
    
    def parse_statement_function(self,statement):
        lhs_expr, rhs_expr = self.parse_assignment(statement)
        lhs_tokens = base.tokenize(lhs_expr)
        name = lhs_tokens.pop(0)
        if not name.isidentifier():
            raise error.SyntaxError("expected identifier")
        if not lhs_tokens.pop(0) == "(":
            raise error.SyntaxError("expected '('")
        args, consumed_tokens = base.get_top_level_operands(lhs_tokens)
        for arg in args:
            if not arg.isidentifier():
                raise error.SyntaxError("expected identifier")
        for i in range(0,consumed_tokens): lhs_tokens.pop(0)
        if not lhs_tokens.pop(0) == ")":
            raise error.SyntaxError("expected ')'")
        return (name, args, rhs_expr)
    
    def parse_parameter_statement(self,statement):
        """:return: List of tuples consisting of parameter name and assigned value.
        ```
        PARAMETER( a = 5, b = 3)
        PARAMETER b = 5, b = 3   ! legacy version
        ```
        will both result in :
        [('a','5'),('b','3')]
        """
        tokens = base.tokenize(statement)
        result = []
        if tokens.pop(0).lower() != "parameter":
            raise error.SyntaxError("expected 'parameter'")
        if tokens[0] == "(":
            # modern variant
            tokens.pop(0)
            variables_raw, num_consumed  = base.get_top_level_operands(tokens)
            tokens = tokens[num_consumed:]
            if tokens.pop(0) != ")":
                raise error.SyntaxError("expected ')'")
        elif tokens[0].isidentifier():
            variables_raw, num_consumed  = base.get_top_level_operands(tokens)
            tokens = tokens[num_consumed:]
        else:
            raise error.SyntaxError("expected '(' or identifier")
        base.check_if_all_tokens_are_blank(tokens)
        if not len(variables_raw):
            raise error.SyntaxError("expected at least one assignment expression")
        for expr in variables_raw: 
            lhs,rhs = self.parse_assignment(expr)
            if not lhs.isidentifier():
                raise error.SyntaxError("expected identifier")
            result.append((lhs,rhs))
        return result

    # rules
    @__no_parse_result
    def is_declaration(self,tokens):
        """No 'function' must be in the tokens.
        """
        return (tokens[0].lower() in ["type","integer","real","complex","logical","character"]
               or base.compare_ignore_case(tokens[0:2],["double","precision"]))
    
    @__no_parse_result
    def is_blank_line(self,statement):
        return not len(statement.strip())
    
    @__no_parse_result
    def is_fortran_directive(self,statement,modern_fortran):
        """If the statement is a directive."""
        return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:2] == "!$" 
           or (not modern_fortran and statement[0:2].lower() in ["c$","*$"]))
    
    @__no_parse_result
    def is_cuda_fortran_conditional_code(self,statement,modern_fortran):
        """If the statement is a directive."""
        return len(statement) > 2 and (modern_fortran and statement.lstrip()[0:5].lower() == "!@cuf" 
           or (not modern_fortran and statement[0:5].lower() in ["c@cuf","*@cuf"]))
    
    @__no_parse_result
    def is_fortran_comment(self,statement,modern_fortran):
        """If the statement is a directive.
        :note: Assumes that `is_cuda_fortran_conditional_code` and 
               `is_fortran_directive` are checked before.
        """
        return len(statement) and (modern_fortran and statement.lstrip()[0] == "!" 
           or (not modern_fortran and statement[0].lower() in "c*"))
    
    
    @__no_parse_result
    def is_cpp_directive(self,statement):
        return statement[0] == "#"
    
    @__no_parse_result
    def is_fortran_offload_region_directive(self,tokens):
        return\
            base.compare_ignore_case(tokens[1:3],["acc","serial"]) or\
            base.compare_ignore_case(tokens[1:3],["acc","parallel"]) or\
            base.compare_ignore_case(tokens[1:3],["acc","kernels"])
    
    @__no_parse_result
    def is_fortran_offload_region_plus_loop_directive(self,tokens):
        return\
            base.compare_ignore_case(tokens[1:4],["cuf","kernel","do"])  or\
            base.compare_ignore_case(tokens[1:4],["acc","parallel","loop"]) or\
            base.compare_ignore_case(tokens[1:4],["acc","kernels","loop"]) or\
            base.compare_ignore_case(tokens[1:4],["acc","kernels","loop"])
    
    @__no_parse_result
    def is_fortran_offload_loop_directive(self,tokens):
        return base.compare_ignore_case(tokens[1:3],["acc","loop"])
    
    
    @__no_parse_result
    def is_derived_type_member_access(self,tokens):
        return len(base.get_top_level_operands(tokens,separators=["%"])[0]) > 1
    
    @__no_parse_result
    def is_assignment(self,tokens):
        """:return if the statement tokens can be split into two
                   parts w.r.t. "=" and the left-hand operand
                   is an lvalue expression.
        """
        #assert not is_declaration_(tokens)
        #assert not is_do_(tokens)
        if "=" in tokens:
            operands,_ = base.get_top_level_operands(tokens,separators=["="],
                                                join_operand_tokens=False)
            if len(operands) == 2:
                lhs = operands[0]
                try:
                    self.parse_lvalue(lhs)
                    return True
                except:
                    return False
            else:
                return False
        else:
            return False
    
    @__no_parse_result
    def is_pointer_assignment(self,tokens):
        #assert not is_ignored_statement_(tokens)
        #assert not is_declaration_(tokens)
        return len(base.get_top_level_operands(tokens,separators=["=>"])[0]) == 2
    
    @__no_parse_result
    def is_subroutine_call(self,tokens):
        return tokens[0] == "call" and tokens[1].isidentifier()
    

    @__produces_parse_result
    def is_interface(self,tokens):
        try:
            self.__parse_result = self.parse_interface_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_select_case(self,tokens):
        try:
            self.__parse_result = self.parse_select_case_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_case(self,tokens):
        try:
            self.__parse_result = self.parse_case_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__no_parse_result
    def is_case_default(self,tokens):
        try:
            self.parse_case_default_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_block(self,tokens):
        """:note: we assume single-line if have been
        transformed in preprocessing step."""
        try:
            self.__parse_result = self.parse_block_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_if_then(self,tokens):
        """:note: we assume single-line if have been
        transformed in preprocessing step."""
        try:
            self.__parse_result = self.parse_if_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_else_if_then(self,tokens):
        try:
            self.__parse_result = self.parse_else_if_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__no_parse_result
    def is_else(self,tokens):
        try:
            self.__parse_result = self.parse_else_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__no_parse_result
    def is_continue(self,tokens):
        if tokens[0].lower() == "continue":
            return True
        elif tokens[1].lower() == "continue":
            try:
                int(tokens[0])
                return True
            except:
                return False 
    
    @__produces_parse_result
    def is_cycle(self,tokens):
        try:
            self.__parse_result = self.parse_cycle_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_exit(self,tokens):
        try:
            self.__parse_result = self.parse_exit_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_unconditional_goto(self,tokens):
        try:
            self.__parse_result = self.parse_unconditional_goto_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_computed_goto(self,tokens):
        try:
            self.__parse_result = self.parse_computed_goto_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
   
    @__produces_parse_result
    def is_assigned_goto(self,tokens):
        try:
            self.__parse_result = self.parse_assigned_goto_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_do(self,tokens):
        try:
            self.__parse_result = self.parse_do_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_where(self,tokens):
        try:
            self.__parse_result = self.parse_where_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_else_where(self,tokens):
        try:
            self.__parse_result = self.parse_else_where_statement(tokens)
            return True
        except (error.SyntaxError):
            return False
    
    @__produces_parse_result
    def is_end(self,tokens,kind=None):
        try:
            self.__parse_result = self.parse_end_statement(tokens,kind)
            return True
        except (error.SyntaxError):
            return False
