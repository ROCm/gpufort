# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#from translator_base import *
import pyparsing
import ast

from .. import opts
from .. import conv
from . import base
from . import grammar


def flatten_arithmetic_expression(expr, converter=base.make_c_str):

    def descend(element, depth=0):
        term = ""
        if isinstance(element, pyparsing.ParseResults):
            for el in element:
                term += "(" if isinstance(el, pyparsing.ParseResults) else ""
                term += descend(el, depth)
                term += ")" if isinstance(el, pyparsing.ParseResults) else ""
            return term
        else:
            return converter(element)

    return descend(expr)


class TTSimpleToken(base.TTNode):

    def _assign_fields(self, tokens):
        self._text = " ".join(tokens)

    def c_str(self):
        return "{};".format(self._text.lower())

    def f_str(self):
        return str(self._text)


class TTReturn(base.TTNode):

    def _assign_fields(self, tokens):
        self._result_name = ""

    def c_str(self):
        if len(self._result_name):
            return "return " + self._result_name + ";"
        else:
            return "return;"

    def f_str(self):
        return "return"


class TTCommentedOut(base.TTNode):

    def _assign_fields(self, tokens):
        self._text = " ".join(tokens)

    def c_str(self):
        return "// {}\n".format(self._text)


class TTIgnore(base.TTNode):

    def c_str(self):
        return ""


class TTLogical(base.TTNode):

    def _assign_fields(self, tokens):
        self._value = tokens[0]

    def c_str(self):
        return "true" if self._value.lower() == ".true." else "false"

    def f_str(self):
        return self._value


class TTNumber(base.TTNode):

    def _assign_fields(self, tokens):
        self._value = tokens[0]

    def is_real(self, kind=None):
        """:return: If the number is a real (of a certain kind)."""
        is_real = "." in self._value
        if is_real and kind != None:
            if kind in fortran_type_2_bytes_map["real"]:
                parts = self._value.split("_")
                has_exponent_e = "e" in self._value
                has_exponent_d = "d" in self._value
                has_exponent = has_exponent_e or has_exponent_d
                default_real_bytes = fortran_type_2_bytes_map["real"][
                    ""].strip()
                kind_bytes = fortran_type_2_bytes_map["real"][kind].strip()
                if len(parts) == 1: # no suffix
                    cond = False
                    cond = cond or (has_exponent_d and kind_bytes == "8")
                    cond = cond or (has_exponent_e and
                                    kind_bytes == default_real_bytes)
                    cond = cond or (not has_exponent and
                                    kind_bytes == default_real_bytes)
                    return cond
                elif len(parts) == 2: # suffix
                    suffix = parts[1]
                    if suffix in fortran_type_2_bytes_map["real"]:
                        suffix_bytes = fortran_type_2_bytes_map["real"][
                            suffix].strip()
                        return kind_bytes == suffix_bytes
                    else:
                        util.logging.log_error(opts.log_prefix,"TTNumber.is_real",\
                          "no number of bytes found for suffix '{}' in 'translator.fortran_type_2_bytes_map[\"real\"]'".format(suffix))
                        sys.exit(2) # TODO error code
            else:
                util.logging.log_error(opts.log_prefix,"TTNumber.is_real",\
                  "no number of bytes found for kind '{}' in 'translator.fortran_type_2_bytes_map[\"real\"]'".format(kind))
                sys.exit(2) # TODO error code
        else:
            return is_real

    def is_integer(self, kind=None):
        """:return: If the number is an integer (of a certain kind)."""
        is_integer = not "." in self._value
        if is_integer and kind != None:
            if kind in opts.fortran_type_2_bytes_map["integer"]:
                parts = self._value.split("_")
                has_exponent_e = "e" in self._value
                has_exponent_d = "d" in self._value
                has_exponent = has_exponent_e or has_exponent_d
                default_integer_bytes = opts.fortran_type_2_bytes_map[
                    "integer"][""].strip()
                kind_bytes = opts.fortran_type_2_bytes_map["integer"][
                    kind].strip()
                if len(parts) == 1: # no suffix
                    return kind_bytes == default_integer_bytes
                elif len(parts) == 2: # suffix
                    suffix = parts[1]
                    if suffix in opts.fortran_type_2_bytes_map["integer"]:
                        suffix_bytes = opts.fortran_type_2_bytes_map[
                            "integer"][suffix].strip()
                        return kind_bytes == suffix_bytes
                    else:
                        util.logging.log_error(opts.log_prefix,"TTNumber.is_integer",\
                          "no number of bytes found for suffix '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(suffix))
                        sys.exit(2) # TODO error code
            else:
                util.logging.log_error(opts.log_prefix,"TTNumber.is_integer",\
                  "no number of bytes found for kind '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(kind))
                sys.exit(2) # TODO error code
        else:
            return is_integer

    def c_str(self):
        parts = self._value.split("_")
        value = parts[0].replace("d", "e")
        if self.is_real("4"):
            return value + "f"
        elif self.is_integer("8"):
            return value + "l"
        else:
            return value

    def f_str(self):
        return self._value


class TTIdentifier(base.TTNode):

    def _assign_fields(self, tokens):
        self._name = tokens[0]

    def f_str(self):
        return str(self._name)

    def c_str(self):
        return self.f_str()


class TTFunctionCallOrTensorAccess(base.TTNode):

    def _assign_fields(self, tokens):
        self._name = tokens[0]
        self._args = tokens[1]
        self._is_tensor_access = base.Unknown3

    def matrix_range_args(self):
        """
        Returns all matrix range args in the order of their appeareance.
        """
        return base.find_all(self._args, searched_type=TTMatrixRange)

    def has_matrix_range_args(self):
        """
        If any matrix range args are present in the argument list.
        """
        return not base.find_first(self._args,
                                   searched_type=TTMatrixRange) is None

    def __guess_it_is_function(self):
        """ 
        Tries to determine if the whole expression
        is function or not if no other hints are given
        """
        name = base.make_c_str(self._name).lower()
        return len(self._args) == 0 or\
          name in opts.gpufort_cpp_symbols or\
          name in ALL_HOST_ROUTINES or\
          name in ALL_DEVICE_ROUTINES

    def is_tensor(self):
        if self._is_tensor_access == base.True3:
            return True
        elif self._is_tensor_access == base.False3:
            return False
        else:
            return self.has_matrix_range_args() or\
                   not self.__guess_it_is_function()

    def __name_c_str(self):
        raw_name = base.make_c_str(self._name).lower()
        num_args = len(self._args)
        if raw_name == "sign":
            return "copysign"
        elif raw_name in ["max", "amax1"]:
            return "max" + str(num_args)
        elif raw_name in ["min", "amin1"]:
            return "min" + str(num_args)
        return raw_name

    def c_str(self):
        name = self.__name_c_str()
        if (not opts.fortran_style_tensor_access) and self.is_tensor():
            return "{0}[_idx_{0}({1})]".format(name, ",".join([
                base.make_c_str(s) for s in self._args
            ])) # Fortran identifiers cannot start with "_"
        else:
            return "{}({})".format(
                name, ",".join([base.make_c_str(s) for s in self._args]))

    def f_str(self):
        name = base.make_f_str(self._name)
        return "{0}({1})".format(
            name, ",".join([base.make_f_str(s) for s in self._args]))


class IValue:

    def is_identifier(self):
        return type(self._value) is TTIdentifier

    def name(self):
        return self._value.f_str()

    def has_matrix_range_args(self):
        """
        Functionality for scanner.
        """
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return self._value.has_matrix_range_args()
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.last_element_has_matrix_range_args()
        else:
            return False

    def matrix_range_args(self):
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return self._value.matrix_range_args()
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.last_element_matrix_range_args()
        else:
            return []


class TTRValue(base.TTNode, IValue):

    def _assign_fields(self, tokens):
        self._sign = tokens[0]
        self._value = tokens[1]
        self._reduction_index = ""
        #print("{0}: {1}".format(self.c_str(),self.location))
    def f_str(self):
        return self._sign + base.make_f_str(self._value)

    def c_str(self):
        result = self._sign + base.make_c_str(self._value)
        if len(self._reduction_index):
            result += "[{idx}]".format(idx=self._reduction_index)
        return result.lower()


class TTLValue(base.TTNode, IValue):

    def _assign_fields(self, tokens):
        self._value = tokens[0]
        self._reduction_index = ""
        #print("{0}: {1}".format(self.c_str(),self.location))
    def f_str(self):
        return base.make_f_str(self._value)

    def c_str(self):
        result = base.make_c_str(self._value)
        if len(self._reduction_index):
            result += "[{idx}]".format(idx=self._reduction_index)
        return result


class TTSizeInquiry(base.TTNode):
    """
    Translator tree node for size inquiry function.
    """

    def _assign_fields(self, tokens):
        self._ref, self._dim, self._kind = tokens

    def c_str(self):
        """
        :return: number of elements per array dimension, if the dimension
                 is specified as argument.
                 Utilizes the <array>_n<dim> and <array>_lb<dim> arguments that
                 are passed as argument of the extracted kernels.
        :note: only the case where <dim> is specified as integer literal is handled by this function.
        """
        if type(self._dim) is TTNumber:
            return base.make_c_str(self._ref) + "_n" + base.make_c_str(
                self._dim)
        else:
            prefix = "size"
            return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"

    def f_str(self):
        prefix = "size"
        result = prefix + "(" + base.make_f_str(self._ref)
        if self._dim != None:
            result += "," + base.make_f_str(self._dim)
        if self._kind != None:
            result += "," + base.make_f_str(self._kind)
        return result + ")"


class TTLboundInquiry(base.TTNode):
    """
    Translator tree node for lbound inquiry function.
    """

    def _assign_fields(self, tokens):
        self._ref, self._dim, self._kind = tokens

    def c_str(self):
        """
        :return: lower bound per array dimension, if the dimension argument is specified as integer literal.
                 Utilizes the <array>_n<dim> and <array>_lb<dim> arguments that
                 are passed as argument of the extracted kernels.
        :note:   only the case where <dim> is specified as integer literal is handled by this function.
        """
        if type(self._dim) is TTNumber:
            return base.make_c_str(self._ref) + "_lb" + base.make_c_str(
                self._dim)
        else:
            prefix = "lbound"
            return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"

    def f_str(self):
        prefix = "lbound"
        result = prefix + "(" + base.make_f_str(self._ref)
        if self._dim != None:
            result += "," + base.make_f_str(self._dim)
        if self._kind != None:
            result += "," + base.make_f_str(self._kind)
        return result + ")"


class TTUboundInquiry(base.TTNode):
    """
    Translator tree node for ubound inquiry function.
    """

    def _assign_fields(self, tokens):
        self._ref, self._dim, self._kind = tokens

    def c_str(self):
        """
        :return: upper bound per array dimension, if the dimension argument is specified as integer literal.
                 Utilizes the <array>_n<dim> and <array>_lb<dim> arguments that
                 are passed as argument of the extracted kernels.
        :note:   only the case where <dim> is specified as integer literal is handled by this function.
        """
        if type(self._dim) is TTNumber:
            return "({0}_lb{1} + {0}_n{1} - 1)".format(
                base.make_c_str(self._ref), base.make_c_str(self._dim))
        else:
            prefix = "ubound"
            return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"

    def f_str(self):
        prefix = "ubound"
        result = prefix + "(" + base.make_f_str(self._ref)
        if self._dim != None:
            result += "," + base.make_f_str(self._dim)
        if self._kind != None:
            result += "," + base.make_f_str(self._kind)
        return result + ")"


class TTConvertToExtractReal(base.TTNode):

    def _assign_fields(self, tokens):
        self._ref, self._kind = tokens

    def c_str(self):
        c_type = conv.convert_to_c_type("real", self._kind).replace(
            " ", "_") # TODO check if his anything else than double or float
        return "make_{1}({0})".format(
            base.make_c_str(self._ref),
            c_type) # rely on C++ compiler to make the correct type conversion

    def f_str(self):
        result = "REAL({0}".format(base.make_f_str(self._ref))
        if not self._kind is None:
            result += ",kind={0}".format(base.make_f_str(self._kind))
        return result + ")"


class TTConvertToDouble(base.TTNode):

    def _assign_fields(self, tokens):
        self._ref, self._kind = tokens

    def c_str(self):
        return "make_double({0})".format(
            base.make_c_str(self._ref)
        ) # rely on C++ compiler to make the correct type conversion

    def f_str(self):
        return "DBLE({0})".format(
            base.make_f_str(self._ref)
        ) # rely on C++ compiler to make the correct type conversion


class TTConvertToComplex(base.TTNode):

    def _assign_fields(self, tokens):
        self._x, self._y, self._kind = tokens

    def c_str(self):
        c_type = conv.convert_to_c_type("complex",
                                        self._kind,
                                        default=None,
                                        float_complex="float_complex",
                                        double_complex="double_complex")
        return "make_{2}({0}, {1})".format(base.make_c_str(self._x),
                                           base.make_c_str(self._y), c_type)

    def f_str(self):
        result = "CMPLX({0},{1}".format(base.make_f_str(self._x),
                                        base.make_f_str(self._y))
        if not self._kind is None:
            result += ",kind={0}".format(base.make_f_str(self._kind))
        return result + ")"


class TTConvertToDoubleComplex(base.TTNode):

    def _assign_fields(self, tokens):
        self._x, self._y, self._kind = tokens

    def c_str(self):
        c_type = "double_complex"
        return "make_{2}({0}, {1})".format(base.make_c_str(self._x),
                                           base.make_c_str(self._y), c_type)

    def f_str(self):
        result = "DCMPLX({0},{1}".format(base.make_f_str(self._x),
                                         base.make_f_str(self._y))
        return result + ")"


class TTExtractImag(base.TTNode):

    def _assign_fields(self, tokens):
        self._ref, self._kind = tokens

    def c_str(self):
        return "{0}._y".format(base.make_c_str(self._ref))


class TTConjugate(base.TTNode):

    def _assign_fields(self, tokens):
        self._ref, self._kind = tokens

    def c_str(self):
        return "conj({0})".format(base.make_c_str(self._ref))


class TTDerivedTypeMember(base.TTNode):

    def _assign_fields(self, tokens):
        self._type, self._element = tokens
        #print(self._type)
    def last_element_matrix_range_args(self):
        """
        Returns all matrix range args in the order of their appeareance.
        """
        result = []
        current = self._element
        while type(current) is TTDerivedTypeMember:
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.matrix_range_args()
        else:
            return []

    def last_element_has_matrix_range_args(self):
        """
        If any matrix range args are present in the argument list.
        """
        result = []
        current = self._element
        while type(current) is TTDerivedTypeMember:
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.has_matrix_range_args()
        else:
            return False

    def c_str(self):
        return base.make_c_str(self._type) + "." + base.make_c_str(
            self._element)

    def f_str(self):
        return base.make_f_str(self._type) + "%" + base.make_f_str(
            self._element)


class TTSubroutineCall(base.TTNode):

    def _assign_fields(self, tokens):
        self._subroutine = tokens[0]

    def c_str(self):
        self._subroutine._is_tensor_access = base.False3
        return self._subroutine.c_str() + ";"


class TTOperator(base.TTNode):

    def _assign_fields(self, tokens):
        self._name = tokens[0]

    def c_str(self):
        f2c = {
            ".eq.": "==",
            "/=": "!=",
            ".ne.": "!=",
            ".neqv.": "!=",
            ".lt.": "<",
            ".gt.": ">",
            ".le.": "<=",
            ".ge.": ">=",
            ".and.": "&",
            ".or.": "|",
            ".xor.": "^",
            ".not.": "!",
            ".eqv.": "==",
        }
        return f2c.get(self._name.lower(), self._name)

    def f_str(self):
        return str(self._name)


class TTArithmeticExpression(base.TTNode):

    def _assign_fields(self, tokens):
        self._expr = tokens

    def children(self):
        return [self._expr]

    def c_str(self):
        return flatten_arithmetic_expression(self._expr)

    def f_str(self):
        return flatten_arithmetic_expression(self._expr, base.make_f_str)


class TTComplexArithmeticExpression(base.TTNode):

    def _assign_fields(self, tokens):
        self._real, self._imag = tokens[0]

    def children(self):
        return [self._real, self._imag]

    def c_str(self):
        return "make_hip_complex({real},{imag})".format(\
                real=flatten_arithmetic_expression(self._real,base.make_c_str),\
                imag=flatten_arithmetic_expression(self._imag,base.make_c_str))

    def f_str(self):
        return "({real},{imag})".format(\
                real=flatten_arithmetic_expression(self._real,base.make_f_str),\
                imag=flatten_arithmetic_expression(self._imag,base.make_f_str))


class TTPower(base.TTNode):

    def _assign_fields(self, tokens):
        self._base, self._exp = tokens

    def children(self):
        return [self._base, self._exp]

    def gpufort_f_str(self, scope=None):
        sign = ""
        base = self._base
        if type(self._base) is TTRValue:
            base = self._base._value
            sign = self._base._sign
        return "{sign}__pow({base},{exp})".format(\
            sign=base.make_f_str(sign),base=base.make_f_str(base),
            exp=base.make_f_str(self._exp))

    __str__ = gpufort_f_str

    def f_str(self):
        return "({base})**({exp})".format(\
            base=base.make_c_str(self._base),exp=base.make_c_str(self._exp))


class TTAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def convert_to_do_loop_nest_if_necessary(self):
        """
        - step through bounds
        - if type is range
          - get upper bound
	    - not found: use a_lb<i>+a_n<i>-1
	  - get lower bound
	     - not found: use a_lb<i>
        """
        name = self._lhs.name()
        loop_nest_start = ""
        loop_nest_end = ""
        original_f_str = self.f_str()
        # TODO fix should be moved into parser preprocessing stage
        for i, matrix_range in enumerate(self._lhs.matrix_range_args()):
            loop_var_name = "_" + chr(
                ord('a') + i) # Fortran names cannot start with "_"
            matrix_range.set_loop_var(loop_var_name)
            lbound = matrix_range.lBound()
            if not len(lbound):
                lbound = "{name}_lb{i}".format(name=name, i=i + 1)
            ubound = matrix_range.uBound()
            if not len(ubound):
                ubound = "({lbound} + {name}_n{i} - 1)".format(lbound=lbound,
                                                               name=name,
                                                               i=i + 1)
            stride = matrix_range.stride()
            if not len(stride):
                stride = "1"
            loop_nest_start = "do {var}={lb},{ub},{step}\n".format(
                var=loop_var_name, lb=lbound, ub=ubound,
                step=stride) + loop_nest_start
            loop_nest_end = "end do\n" + loop_nest_end
        if len(loop_nest_start):
            # TODO put into options
            for rvalue in base.find_all(self._rhs, searched_type=TTRValue):
                for i, matrix_range in enumerate(rvalue.matrix_range_args()):
                    loop_var_name = "_" + chr(
                        ord('a') + i) # Fortran names cannot start with "_"
                    matrix_range.set_loop_var(loop_var_name)
            loop_nest_start += "! TODO(gpufort) please check if this conversion was done correctly\n"
            loop_nest_start += "! original: {}\n".format(original_f_str)
            result = loop_nest_start + self.f_str() + "\n" + loop_nest_end
            return annotated_do_loop.parseString(result)[0]
        else:
            return self

    def c_str(self, unpack=False):
        """
        :param unpack: Converts this assignment to a do loop nest if any
        matrix ranges ([...]:[...]) are found.
        """
        if unpack:
            return self.convert_to_do_loop_nest_if_necessary(
            ).c_str()
        else:
            return  self._lhs.c_str(
            ) + "=" + flatten_arithmetic_expression(self._rhs) + ";\n"

    def f_str(self, unpack=False):
        """
        """
        if unpack:
            return self.convert_to_do_loop_nest_if_necessary().f_str()
        else:
            return self._lhs.f_str() + "=" + flatten_arithmetic_expression(
                self._rhs, converter=base.make_f_str) + ";\n"


class TTRange(base.TTNode):

    def _assign_fields(self, tokens):
        self._begin, self._end = tokens
        del self._tokens

    def f_str(self):
        return self._begin.f_str() + ":" + self._end.f_str()

    def c_str(self):
        return self.f_str()


class TTComplexAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def c_str(self):
        """
        Expand the complex assignment.
        """
        result = ""
        result += "{}.x = {};\n".format(base.make_c_str(self._lhs),
                                        base.make_c_str(self._rhs._real))
        result += "{}.y = {};\n".format(base.make_c_str(self._lhs),
                                        base.make_c_str(self._rhs._imag))
        return result


class TTMatrixAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def c_str(self):
        """
        Expand the matrix assignment.
        User still has to fix the ranges manually. 
        """
        result = "// TODO: fix ranges"
        for expression in self._rhs:
            result += base.make_c_str(
                self._lhs) + argument + "=" + flatten_arithmetic_expression(
                    expression) + ";\n"
        return result


class TTIntentQualifier(base.TTNode):

    def _assign_fields(self, tokens):
        #print("found intent_qualifier")
        #print(tokens[0])
        self._intent = tokens[0][0]

    def c_str(self):
        #print(self._intent)
        return "const" if self._intent.lower() == "in" else ""

    def f_str(self):
        return "intent({0})".format(self._intent)


class TTMatrixRange(base.TTNode):

    def _assign_fields(self, tokens):
        self._lbound, self._ubound, self._stride = tokens
        self._loop_var = ""

    def set_loop_var(self, name):
        self._loop_var = name

    def lBound(self, converter=base.make_c_str):
        return converter(self._lbound)

    def uBound(self, converter=base.make_c_str):
        return converter(self._ubound)

    def unspecified_l_bound(self):
        return not len(self.lBound())

    def unspecified_u_bound(self):
        return not len(self.uBound())

    def stride(self, converter=base.make_c_str):
        return converter(self._stride)

    def size(self, converter=base.make_c_str):
        result = "{1} - ({0}) + 1".format(converter(self._lbound),
                                          converter(self._ubound))
        try:
            result = str(ast.literal_eval(result))
        except:
            try:
                result = " - ({0}) + 1".format(converter(self._lbound))
                result = str(ast.literal_eval(result))
                if result == "0":
                    result = converter(self._ubound)
                else:
                    result = "{1}{0}".format(result, converter(self._ubound))
            except:
                pass
        return result

    def c_str(self):
        #return self.size(base.make_c_str)
        if len(self._loop_var):
            return self._loop_var # if a do loop is wrapped around this range
        else:
            return "/*TODO fix this BEGIN*/{0}/*fix END*/".format(self.f_str())

    def f_str(self):
        if len(self._loop_var):
            return self._loop_var # if a do loop is wrapped around this range
        else:
            result = ""
            if not self._lbound is None:
                result += base.make_f_str(self._lbound)
            result += ":"
            if not self._ubound is None:
                result += base.make_f_str(self._ubound)
            if not self._stride is None:
                result += ":" + base.make_f_str(self._stride)
            return result


class TTBounds(base.TTNode):
    """
    Spawned from grammar:
    
    ```python
    dimension_list  = delimitedList(dimension_value)
    ```
 
    where
    
    ```python 
    dimension_value = ( matrix_range | arithmetic_expression | Literal("*"))
    ```
    """

    def _assign_fields(self, tokens):
        self._bounds = []
        if tokens != None:
            self._bounds = tokens[0]

    def has_unspecified_bounds(self, converter=base.make_c_str):
        """
        Typically the case if the declaration variable has
        a ALLOCATABLE or POINTER qualifier.
        """
        result = False
        for dim in self._bounds:
            result |= type(dim) is str and dim == "*"
            result |= type(dim) in [
                TTMatrixRange
            ] and (dim.unspecified_l_bound() or dim.unspecified_u_bound())
        return result

    def rank(self):
        """
        :return: the rank, i.e. the number of dimensions.
        """
        return len(self._bounds)

    def specified_bounds(self, converter=base.make_c_str):
        """
        :return: list that contains a list [lower, upper] per specified dimension of the array.
        Note that in Fortran the fastest running ("stride(1)")
        index is on the left in expressions `(i0,i1,..,in)`, while in C the fastest
        running index is on the right in expressions `[in][in-1]...[i0]`.
        """
        result = []
        for el in self._bounds:
            if type(el) is TTArithmeticExpression:
                result.append(["1", converter(el)])
            elif type(el) is TTMatrixRange:
                result.append([el.lBound(converter), el.uBound(converter)])
            elif el in [":", "*"]:
                pass
        return result

    def specified_counts(self, converter=base.make_c_str):
        """
        Used for mallocs. These are just the counts.
        (Number of bytes requires knowledge of datatype too.)
        """
        result = [
            "{0}".format(el[1]) if el[0] == "1" else "{1} - ({0}) + 1".format(
                el[0], el[1]) for el in self.specified_bounds(converter)
        ]
        for i, el in enumerate(result):
            try:
                result[i] = str(ast.literal_eval(el))
            except:
                result[i] = el
        return result

    def specified_lower_bounds(self, converter=base.make_c_str):
        """
        Used for mallocs. These are just the counts.
        (Number of bytes requires knowledge of datatype too.)
        """
        result = [el[0] for el in self.specified_bounds(converter)]
        return result

    def size(self, bytes_per_element="1", converter=base.make_c_str):
        """
        Used for mallocs. These are just the counts.
        (Number of bytes requires knowledge of datatype too.)
        """
        result = "*".join(
            ["({0})".format(dim) for dim in self.specified_counts(converter)])
        if bytes_per_element != "1":
            result = "(" + bytes_per_element + ")*" + result
        if converter is base.make_f_str:
            result = "1_8*" + result
        return result

    def bound_var_names(self, variable_name, converter=base.make_c_str):
        """
        Used for function arguments and declarations
        """
        result = [
            "{0}_n{1}".format(variable_name, i + 1)
            for i in range(0, len(self._bounds))
        ]
        result += [
            "{0}_lb{1}".format(variable_name, i + 1)
            for i in range(0, len(self._bounds))
        ]
        return result

    def bound_var_assignments(self,
                                   variable_name,
                                   converter=base.make_c_str):
        """
        Used for assigning values to the bound variables. Can in general not be put in
        declaration as rhs may not be constant.
        """
        counts = self.specified_counts(converter)
        lower_bounds = self.specified_lower_bounds(converter)
        result = [
            "{0}_n{1} = {2}".format(variable_name, i + 1, val)
            for i, val in enumerate(counts)
        ]
        result += [
            "{0}_lb{1} = {2}".format(variable_name, i + 1, val)
            for i, val in enumerate(lower_bounds)
        ]
        return result

    def index_str(self, variable_name, use_place_holders=False, macro_args=[]):
        """
        Linearises the multivariate index required to access the original Fortran array.
        :param variable_name: Name of the array this bound is associated with.
        :type variable_name: str
        :return: an expression for the index (as string) or None if no bounds were specified.
        """
        bounds_specified = not self.has_unspecified_bounds()
        bounds = self.specified_bounds(
            base.make_c_str) if bounds_specified else self._bounds
        lower_bounds = self.specified_lower_bounds(
            base.make_c_str) if bounds_specified else ["1"] * len(
                self._bounds) # TODO
        if bounds_specified:
            counts = self.specified_counts()
        if len(bounds):
            if not len(macro_args):
                macro_args = [chr(ord('a') + i) for i in range(0, len(bounds))]
            assert len(macro_args) == len(bounds)
            if not bounds_specified or use_place_holders:
                bound_args = [
                    "{0}_n{1}".format(variable_name, i + 1)
                    for i in range(0, len(bounds))
                ]
                lower_bound_args = [
                    "{0}_lb{1}".format(variable_name, i + 1)
                    for i in range(0, len(bounds))
                ]
            else:
                bound_args = counts
                lower_bound_args = lower_bounds
            stride = ""
            prod = ""
            index = ""
            for i, mvar in enumerate(macro_args):
                index += "{0}({1}-({2}))".format(stride, mvar,
                                                 lower_bound_args[i])
                prod = "{}{}*".format(prod, bound_args[i])
                stride = "+{0}".format(prod)
            return index
        else:
            return None

    def index_macro_str(self,
                        variable_name,
                        converter=base.make_c_str,
                        use_place_holders=False):
        """
        Linearizes the multivariate index required to access the original Fortran array.
        :param variable_name: Name of the array this bound is associated with.
        :type variable_name: str
        :return: an expression for the index (as string) or None if no bounds were specified.
        """
        bounds_specified = not self.has_unspecified_bounds()
        bounds = list(self.specified_bounds(
            base.make_c_str)) if bounds_specified else self._bounds
        macro_args = [chr(ord('a') + i) for i in range(0, len(bounds))]
        index = self.index_str(variable_name, use_place_holders)
        if converter is base.make_f_str:
            return "#undef {0}\n#define {0}({1}) inc_c_ptr({0}, {2})".format(
                variable_name, ",".join(macro_args), index)
        else:
            return "#undef _idx_{0}\n#define _idx_{0}({1}) ({2})".format(
                variable_name, ",".join(macro_args), index)

    def index_macro_f_str(self, variable_name, use_place_holders=False):
        return self.index_macro_str(variable_name, base.make_f_str,
                                    use_place_holders)

    def index_macro_c_str(self, variable_name, use_place_holders=False):
        return self.index_macro_str(variable_name, base.make_c_str,
                                    use_place_holders)

    def c_str(self):
        """
        Returns '[size(n)][size(n-1)]...[size1]' for all n sizes corresponding to specified bounds
        """
        #print("{0}".format(self._bounds))
        #print("{0}".format(self.specified_counts(base.make_c_str)))
        return "".join(
            ["[{0}]".format(base.make_c_str(el)) for el in self._bounds])

    def f_str(self):
        """
        Returns '(<bound1.f_str()>,<bound2.f_str()>,...,<bound(n).f_str()>)' for all n sizes corresponding to specified bounds
        """
        return "({0})".format(",".join(
            base.make_f_str(el) for el in self._bounds))


class TTDimensionQualifier(base.TTNode):

    def _assign_fields(self, tokens):
        self._bounds = tokens[0][0]

    def c_str(self):
        return base.make_c_str(self._bounds)

    def f_str(self):
        return "dimension{0}".format(base.make_f_str(self._bounds))


class Attributed():

    def get_string_qualifiers(self):
        """
        :param name: lower case string qualifier, i.e. no more complex qualifier such as 'dimension' or 'intent'.
        """
        return [q.lower() for q in self.qualifiers if type(q) is str]

    def has_string_qualifier(self, name):
        """
        :param name: lower case string qualifier, i.e. no more complex qualifier such as 'dimension' or 'intent'.
        """
        result = False
        for q in self.qualifiers:
            result |= type(q) is str and q.lower() == name.lower()
        return result

    def has_dimension(self):
        return len(base.find_all(self.qualifiers, TTDimensionQualifier))

class TTStatement(base.TTNode):
    """
    Mainly used for searching expressions and replacing them in the translator AST.
    One example is converting to colon operations to do loops.
    """

    def _assign_fields(self, tokens):
        self._statement = tokens

    def children(self):
        return [self._statement]

    def c_str(self):
        return base.make_c_str(self._statement)

    def f_str(self):
        return base.make_f_str(self._statement)


class TTIfElseBlock(base.TTContainer):
    def _assign_fields(self, tokens):
        self.indent = "" # container of if/elseif/else branches, so no indent

class TTIfElseIf(base.TTContainer):

    def _assign_fields(self, tokens):
        self._else, self._condition, self.body = tokens

    def children(self):
        return [self._condition, self.body]

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        return "{0}if ({1}) {{\n{2}\n}}".format(\
            self._else,base.make_c_str(self._condition),body_content)


class TTElse(base.TTContainer):

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        return "else {{\n{0}\n}}".format(\
            body_content)


class TTDoWhile(base.TTContainer):

    def _assign_fields(self, tokens):
        self._condition, self.body = tokens

    def children(self):
        return [self._condition, self.body]

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        condition_content = base.make_c_str(self._condition)
        c_str = "while ({0}) {{\n{1}\n}}".format(\
          condition_content,body_content)
        return c_str


## Link actions
#print_statement.setParseAction(TTCommentedOut)
grammar.comment.setParseAction(TTCommentedOut)
grammar.logical.setParseAction(TTLogical)
grammar.integer.setParseAction(TTNumber)
grammar.number.setParseAction(TTNumber)
grammar.l_arith_operator.setParseAction(TTOperator)
#r_arith_operator.setParseAction(TTOperator)
grammar.condition_op.setParseAction(TTOperator)
grammar.identifier.setParseAction(TTIdentifier)
grammar.rvalue.setParseAction(TTRValue)
grammar.lvalue.setParseAction(TTLValue)
grammar.simple_derived_type_member.setParseAction(TTDerivedTypeMember)
grammar.derived_type_elem.setParseAction(TTDerivedTypeMember)
grammar.func_call.setParseAction(TTFunctionCallOrTensorAccess)
grammar.convert_to_extract_real.setParseAction(TTConvertToExtractReal)
grammar.convert_to_double.setParseAction(TTConvertToDouble)
grammar.convert_to_complex.setParseAction(TTConvertToComplex)
grammar.convert_to_double_complex.setParseAction(TTConvertToDoubleComplex)
grammar.extract_imag.setParseAction(TTExtractImag)
grammar.conjugate.setParseAction(TTConjugate)
grammar.conjugate_double_complex.setParseAction(TTConjugate) # same action
grammar.size_inquiry.setParseAction(TTSizeInquiry)
grammar.lbound_inquiry.setParseAction(TTLboundInquiry)
grammar.ubound_inquiry.setParseAction(TTUboundInquiry)
grammar.matrix_range.setParseAction(TTMatrixRange)
grammar.bounds.setParseAction(TTBounds)
grammar.matrix_ranges.setParseAction(TTBounds)
grammar.dimension_qualifier.setParseAction(TTDimensionQualifier)
grammar.intent_qualifier.setParseAction(TTIntentQualifier)
grammar.arithmetic_expression.setParseAction(TTArithmeticExpression)
grammar.arithmetic_logical_expression.setParseAction(TTArithmeticExpression)
grammar.complex_arithmetic_expression.setParseAction(
    TTComplexArithmeticExpression)
grammar.power_value1.setParseAction(TTRValue)
grammar.power.setParseAction(TTPower)
grammar.assignment.setParseAction(TTAssignment)
grammar.matrix_assignment.setParseAction(TTMatrixAssignment)
grammar.complex_assignment.setParseAction(TTComplexAssignment)
# statements
grammar.return_statement.setParseAction(TTReturn)
grammar.fortran_subroutine_call.setParseAction(TTSubroutineCall)
