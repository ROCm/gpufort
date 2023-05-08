
#from translator_base import *
import pyparsing
import ast

from gpufort import util

from .. import opts
from .. import conv
from .. import prepostprocess

from . import base
from . import grammar

import enum

def flatten_arithmetic_expression(expr, converter=base.make_c_str):

    def descend_(current, depth=0):
        if isinstance(current, (list,pyparsing.ParseResults)):
            result = ""
            if (depth > 0): result += "("
            for element in current:
                result += descend_(element, depth+1)
            if (depth > 0): result += ")"
            return result
        elif isinstance(current,(TTArithmeticExpression)):
            return descend_(current._expr,depth) 
        else:
            return converter(current)

    return descend_(expr)


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
        if self._result_name != None and len(self._result_name):
            return "return " + self._result_name + ";"
        else:
            return "return;"

    def f_str(self):
        return "return"

class TTLabel(base.TTNode):

    def _assign_fields(self, tokens):
        self._label = tokens[0]

    def c_str(self):
        return "_{}:".format(self._label)

class TTContinue(base.TTNode):

    def _assign_fields(self, tokens):
        self._result_name = ""
        self._in_loop = True

    def c_str(self):
        if self._in_loop:
            return "continue;"
        elif self._result_name != None and len(self._result_name):
            return "return " + self._result_name + ";"
        else:
            return "return;"

    def f_str(self):
        if self._in_loop:
            return "continue;"
        else:
            return "return"

class TTExit(base.TTNode):

    def _assign_fields(self, tokens):
        self._result_name = ""
        self._in_loop = True

    def c_str(self):
        if self._in_loop:
            return "break;"
        elif self._result_name != None and len(self._result_name):
            return "return " + self._result_name + ";"
        else:
            return "return;"

    def f_str(self):
        if self._in_loop:
            return "break;"
        else:
            return "return"

class TTGoto(base.TTNode):

    def _assign_fields(self, tokens):
        self._label = tokens[0]

    def c_str(self):
        return "goto _{};".format(self._label.rstrip("\n"))

class TTCommentedOut(base.TTNode):

    def _assign_fields(self, tokens):
        self._text = " ".join(tokens)

    def c_str(self):
        return "// {}".format(self._text)


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
            if kind in opts.fortran_type_2_bytes_map["real"]:
                parts = self._value.split("_")
                has_exponent_e = "e" in self._value
                has_exponent_d = "d" in self._value
                has_exponent = has_exponent_e or has_exponent_d
                default_real_bytes = opts.fortran_type_2_bytes_map["real"][
                    ""].strip()
                kind_bytes = opts.fortran_type_2_bytes_map["real"][kind].strip()
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
                    if suffix in opts.fortran_type_2_bytes_map["real"]:
                        suffix_bytes = opts.fortran_type_2_bytes_map["real"][
                            suffix].strip()
                        return kind_bytes == suffix_bytes
                    else:
                        raise util.error.LookupError(\
                          "no number of bytes found for suffix '{}' in 'translator.fortran_type_2_bytes_map[\"real\"]'".format(suffix))
                        sys.exit(2) # TODO error code
            else:
                raise util.error.LookupError(\
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
                        raise ValueError("no number of bytes found for suffix '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(suffix))
            else:
                raise ValueError(\
                  "no number of bytes found for kind '{}' in 'translator.opts.fortran_type_2_bytes_map[\"integer\"]'".format(kind))
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
    
    def __str__(self):
        return "TTNumber(val:"+str(self._value)+")"
    __repr__ = __str__


class TTIdentifier(base.TTNode):

    def _assign_fields(self, tokens):
        self._name = tokens[0]

    def f_str(self):
        return str(self._name)

    def c_str(self):
        return self.f_str()

    def __str__(self):    
        return "TTIdentifier(name:"+str(self._name)+")"
    __repr__ = __str__


class TTFunctionCallOrTensorAccess(base.TTNode):
    class Type(enum.Enum):
        UNKNOWN = 0
        ARRAY_ACCESS = 0
        INTRINSIC_CALL = 1
        FUNCTION_CALL = 2

    def _assign_fields(self, tokens):
        self._name = tokens[0]
        self._args = tokens[1]
        self._type = TTFunctionCallOrTensorAccess.Type.UNKNOWN

    def range_args(self):
        """Returns all range args in the order of their appeareance.
        """
        return base.find_all(self._args, searched_type=TTRange)

    def has_range_args(self):
        """If any range args are present in the argument list.
        """
        return base.find_first(self._args,searched_type=TTRange) != None
    
    def args(self):
        """Returns all args in the order of their appeareance.
        """
        return self._args

    def has_args(self):
        """If any args are present.
        """
        return len(self._args) 

    def __guess_it_is_function(self):
        """ 
        Tries to determine if the whole expression
        is function or not if no other hints are given
        """
        #  TODO hacky old solution, evaluate if still needed
        name = base.make_c_str(self._name).lower()
        return len(self._args) == 0 or\
          name in opts.gpufort_cpp_symbols or\
          name in grammar.ALL_HOST_ROUTINES or\
          name in grammar.ALL_DEVICE_ROUTINES

    def is_tensor(self):
        if self._type == TTFunctionCallOrTensorAccess.Type.ARRAY_ACCESS:
            return True
        elif self._type == TTFunctionCallOrTensorAccess.Type.UNKNOWN:
            return self.has_range_args() or\
                   not self.__guess_it_is_function()
        else:
            return False

    def name_c_str(self):
        name = base.make_c_str(self._name).lower()
        if self.is_tensor():
            return name
        elif self._type == TTFunctionCallOrTensorAccess.Type.INTRINSIC_CALL:
            name = prepostprocess.modernize_fortran_function_name(name)
            if name in [
              "max",
              "min",
              ]:
                num_args = len(self._args)
                return name + str(num_args)
            else:
                return name
        else:
            return name

    def c_str(self):
        name = self.name_c_str()
        return "".join([
            name,
            self._args.c_str(name,
                             self.is_tensor(),
                             opts.fortran_style_tensor_access)])
    
    def f_str(self):
        name = base.make_f_str(self._name)
        return "".join([
            name,
            self._args.f_str()
            ])
    
    def __str__(self):
        return "TTFunctionCallOrTensorAccess(name:"+str(self._name)+",is_tensor:"+str(self.is_tensor())+")"
    __repr__ = __str__

class TTValue(base.TTNode):
    
    def _assign_fields(self, tokens):
        self._value           = tokens[0][0]
        self._reduction_index = None
        self._f_str           = None
   
    def is_identifier(self):
        return isinstance(self._value, TTIdentifier)

    def identifier_part(self,converter=base.make_f_str):
        """
        :return: The identifier part of the expression. In case
                 of a function call/tensor access expression, 
                 excludes the argument list.
                 In case of a derived type, excludes the argument
                 list of the innermost member.
        """
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return converter(self._value._name)
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.identifier_part(converter)
        else:
            return converter(self._value)
    
    def is_function_call(self):
        """
        :note: TTFunctionCallOrTensorAccess instances must be flagged as tensor beforehand.
        """
        #TODO check if detect all arrays in indexer/scope
        # so that we do not need to know function names anymore.
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return not self._value.is_tensor()
        elif type(self._value) is TTDerivedTypeMember:
            # TODO support type bounds routines
            return False
        else:
            return False

    def get_value(self):
        return self._value 

    def name(self):
        return self._value.f_str()
    
    def has_range_args(self):
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return self._value.has_range_args()
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.innermost_member_has_range_args()
        else:
            return False

    def range_args(self):
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return self._value.range_args()
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.innermost_member_range_args()
        else:
            return []
    
    def has_args(self):
        """:return If the value type expression has an argument list. In
                   case of a derived type member, if the inner most derived
                   type member has an argument list.
        """
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return True
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.innermost_member_has_args()
        else:
            return False
    
    def args(self):
        if type(self._value) is TTFunctionCallOrTensorAccess:
            return self._value._args
        elif type(self._value) is TTDerivedTypeMember:
            return self._value.innermost_member_args()
        else:
            return []
    
    def overwrite_f_str(self,f_str):
        self._f_str = f_str
    
    def f_str(self):
        if self._f_str != None:
            return self._f_str
        else:
            return base.make_f_str(self._value)

    def c_str(self):
        result = base.make_c_str(self._value)
        if self._reduction_index != None:
            if opts.fortran_style_tensor_access:
                result += "({idx})".format(idx=self._reduction_index)
            else:
                result += "[{idx}]".format(idx=self._reduction_index)
        return result.lower()

class TTLValue(TTValue):
    def __str__(self):
        return "TTLValue(val:"+str(self._value)+")"
    __repr__ = __str__

class TTRValue(TTValue):
    def __str__(self):
        return "TTRValue(val:"+str(self._value)+")"
    __repr__ = __str__

def _inquiry_str(prefix,ref,dim,kind=""):
    result = prefix + "(" + ref
    if len(dim):
        result += "," + dim
    if len(kind):
        result += "," + kind
    return result + ")"

def _inquiry_c_str(prefix,ref,dim,kind,f_type="integer",c_type_expected="int"):
    """Tries to determine a C type before calling _inquiry_str.
    Wraps the result of the latter function into a typecast
    if the C type does not equal the expected type.
    """
    c_type = conv.convert_to_c_type(f_type,
                                    kind,
                                    default=None)
    result = _inquiry_str(prefix,ref,dim)
    if c_type_expected != c_type:
        return "static_cast<{}>({})".format(c_type,result)
    else:
        return result

class TTSizeInquiry(base.TTNode):
    """Translator tree node for size inquiry function.
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
        if opts.fortran_style_tensor_access:
            return _inquiry_c_str("size",
                                  base.make_c_str(self._ref),
                                  base.make_c_str(self._dim),
                                  base.make_c_str(self._kind))
        else:
            if type(self._dim) is TTNumber:
                return base.make_c_str(self._ref) + "_n" + base.make_c_str(
                    self._dim)
            else:
                prefix = "size"
                return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"
    def f_str(self):
        return _inquiry_str("size",
                            base.make_f_str(self._ref),
                            base.make_f_str(self._dim),
                            base.make_f_str(self._kind))


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
        if opts.fortran_style_tensor_access:
            return _inquiry_c_str("lbound",
                                  base.make_c_str(self._ref),
                                  base.make_c_str(self._dim),
                                  base.make_c_str(self._kind))
        else:
            if type(self._dim) is TTNumber:
                return base.make_c_str(self._ref) + "_lb" + base.make_c_str(
                    self._dim)
            else:
                prefix = "lbound"
                return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"

    def f_str(self):
        return _inquiry_str("lbound",
                            base.make_f_str(self._ref),
                            base.make_f_str(self._dim),
                            base.make_f_str(self._kind))


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
        if opts.fortran_style_tensor_access:
            return _inquiry_c_str("ubound",
                                  base.make_c_str(self._ref),
                                  base.make_c_str(self._dim),
                                  base.make_c_str(self._kind))
        else:
            if type(self._dim) is TTNumber:
                return "({0}_lb{1} + {0}_n{1} - 1)".format(
                    base.make_c_str(self._ref), base.make_c_str(self._dim))
            else:
                prefix = "ubound"
                return "/* " + prefix + "(" + base.make_f_str(self._ref) + ") */"
    def f_str(self):
        return _inquiry_str("ubound",
                            base.make_f_str(self._ref),
                            base.make_f_str(self._dim),
                            base.make_f_str(self._kind))


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
                                        float_complex="hipFloatComplex",
                                        double_complex="hipDoubleComplex")
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
        self._c_str = None

    def identifier_part(self,converter=base.make_f_str):
        result = converter(self._type)
        current = self._element
        while isinstance(current,TTDerivedTypeMember):
            current = current._element
            result += "%"+converter(self._type)
        if isinstance(current,TTFunctionCallOrTensorAccess):
            result += "%"+converter(current._name)
        else: # TTIdentifier
            result += "%"+converter(current)
        return result             

    def innermost_member_range_args(self):
        """Returns all range args in the order of their appeareance.
        """
        result = []
        current = self._element
        while isinstance(current,TTDerivedTypeMember):
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.range_args()
        else:
            return []

    def innermost_member_has_range_args(self):
        """If any range args are present in the argument list.
        """
        result = []
        current = self._element
        while isinstance(current,TTDerivedTypeMember):
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.has_range_args()
        else:
            return False
    
    def innermost_member_has_args(self):
        """If any range args are present in the argument list.
        """
        result = []
        current = self._element
        while isinstance(current,TTDerivedTypeMember):
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.has_args()
        else:
            return False
    
    def innermost_member_args(self):
        """If any range args are present in the argument list.
        """
        result = []
        current = self._element
        while isinstance(current,TTDerivedTypeMember):
            current = current._element
        if type(current) is TTFunctionCallOrTensorAccess:
            return current.args()
        else:
            return []

    def overwrite_c_str(self,expr):
        self._c_str = expr

    def c_str(self):
        if self._c_str == None:
            return base.make_c_str(self._type) + "." + base.make_c_str(
                self._element)
        else:
            return self._c_str

    def f_str(self):
        return base.make_f_str(self._type) + "%" + base.make_f_str(
            self._element)
    
    def __str__(self):
        return "TTDerivedTypeMember(name:"+str(self._type)+"member:"+str(self._element)+")"
    __repr__ = __str__


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
            ".and.": "&&",
            ".or.": "||",
            ".xor.": "^",
            ".not.": "!",
            ".eqv.": "==",
        }
        return f2c.get(self._name.lower(), self._name)

    def f_str(self):
        return str(self._name)


class TTArithmeticExpression(base.TTNode):

    def _assign_fields(self, tokens):
        self._expr = tokens[0]

    def child_nodes(self):
        return [self._expr]

    def c_str(self):
        return flatten_arithmetic_expression(self)

    def f_str(self):
        return flatten_arithmetic_expression(self, base.make_f_str)


class TTComplexArithmeticExpression(base.TTNode):

    def _assign_fields(self, tokens):
        self._real, self._imag = tokens[0]

    def child_nodes(self):
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
        self.base, self.exp = tokens

    def child_nodes(self):
        return [self.base, self.exp]

    def gpufort_f_str(self, scope=None):
        return "__pow({base},{exp})".format(base=base.make_f_str(self.base),
            exp=base.make_f_str(self.exp))

    def __str__(self):
        return self.gpufort_f_str()

    def f_str(self):
        return "({base})**({exp})".format(\
            base=base.make_c_str(self.base),exp=base.make_c_str(self.exp))

class TTAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def c_str(self):
        return self._lhs.c_str() + "=" + flatten_arithmetic_expression(self._rhs) + ";\n"

    def child_nodes(self):
        return [self._lhs, self._rhs]

    def f_str(self):
        return "".join([self._lhs.f_str(),"=",flatten_arithmetic_expression(
            self._rhs, converter=base.make_f_str),"\n"])

class TTComplexAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self._lhs, self._rhs = tokens

    def child_nodes(self):
        return [self._lhs, self._rhs]

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

    def child_nodes(self):
        return [self._lhs, self._rhs]

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

class TTRange(base.TTNode):

    def _assign_fields(self, tokens):
        self._lbound, self._ubound, self._stride = tokens
        self._f_str = None

    def set_loop_var(self, name):
        self._loop_var = name

    def l_bound(self, converter=base.make_c_str):
        return converter(self._lbound)

    def u_bound(self, converter=base.make_c_str):
        return converter(self._ubound)

    def unspecified_l_bound(self):
        return not len(self.l_bound())

    def unspecified_u_bound(self):
        return not len(self.u_bound())

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
        return "/*TODO fix this BEGIN*/{0}/*fix END*/".format(self.f_str())

    def overwrite_f_str(self,f_str):
        self._f_str = f_str
    
    def f_str(self):
        if self._f_str != None:
            return self._f_str
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


class TTArgumentList(base.TTNode):
    def _assign_fields(self, tokens):
        self.items = []
        self.max_rank = -1
        if tokens != None:
            try:
                self.items = tokens[0].asList()
            except AttributeError:
                if isinstance(tokens[0],list):
                    self.items = tokens[0]
                else:
                    raise
            self.max_rank = len(self.items)
        self.__next_idx = 0
    def __len__(self):
        return len(self.items)
    def __iter__(self):
        return iter(self.items)

    def c_str(self):
        return "".join(
            ["[{0}]".format(base.make_c_str(el)) for el in self.items])

    def __max_rank_adjusted_items(self):
        if self.max_rank > 0:
            assert self.max_rank <= len(self.items)
            result = self.items[0:self.max_rank]
        else:
            result = []
        return result

    def c_str(self,name,is_tensor=False,fortran_style_tensor_access=True):
        args = self.__max_rank_adjusted_items()
        if len(args):
            if (not fortran_style_tensor_access and is_tensor):
                return "[_idx_{0}({1})]".format(name, ",".join([
                    base.make_c_str(s) for s in args
                ])) # Fortran identifiers cannot start with "_"
            else:
                return "({})".format(
                    ",".join([base.make_c_str(s) for s in args]))
        else:
            return ""

    def f_str(self):
        args = self.__max_rank_adjusted_items()
        if len(args):
            return "({0})".format(",".join(
                base.make_f_str(el) for el in args))
        else:
            return ""

class TTIfElseBlock(base.TTContainer):
    def _assign_fields(self, tokens):
        self.indent = "" # container of if/elseif/else branches, so no indent

class TTIfElseIf(base.TTContainer):

    def _assign_fields(self, tokens):
        self._else, self._condition, self.body = tokens

    def child_nodes(self):
        return [self._condition, self.body]

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        prefix = self._else+" " if self._else.lower() == "else" else ""
        return "{}if ({}) {{\n{}\n}}".format(\
            prefix,base.make_c_str(self._condition),body_content)


class TTElse(base.TTContainer):

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        return "else {{\n{}\n}}".format(\
            body_content)

class TTSelectCase(base.TTContainer):
    def _assign_fields(self, tokens):
        self.selector = tokens[0]
        self.indent = "" # container of if/elseif/else branches, so no indent
    
    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        return "switch ({}) {{\n{}\n}}".format(\
            base.make_c_str(self.selector),body_content)

class TTCase(base.TTContainer):

    def _assign_fields(self, tokens):
        self.cases, self.body = tokens

    def child_nodes(self):
        return [self.cases, self.body]

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        result = ""
        for case in self.cases:
            result += "case ("+base.make_c_str(case)+"):\n"
        result += body_content + "\n  break;"
        return result

class TTCaseDefault(base.TTContainer):

    def _assign_fields(self, tokens):
        self.body = tokens[0]

    def child_nodes(self):
        return [self.body]

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        return "default:\n{}\n  break;".format(body_content)

class TTDoWhile(base.TTContainer):

    def _assign_fields(self, tokens):
        self._condition, self.body = tokens

    def child_nodes(self):
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
grammar.l_arith_operator_1.setParseAction(TTOperator)
grammar.l_arith_operator_2.setParseAction(TTOperator)
#r_arith_operator.setParseAction(TTOperator)
grammar.condition_op_1.setParseAction(TTOperator)
grammar.condition_op_2.setParseAction(TTOperator)
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
grammar.array_range.setParseAction(TTRange)
grammar.arglist.setParseAction(TTArgumentList)
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
grammar.fortran_subroutine_call.setParseAction(TTSubroutineCall)
