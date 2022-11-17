# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

# third-party modules
#import cPyparsing as pyp # todo: unfortunately, evhub/cpyparsing seems to generate different parse tree
# underlying pyparsing implementation might be too oudated (is from "python2.4 times").
# Substantial performance gains might make it worth it to investigate this further.

try:
    import cPyparsing as pyp
except:
    import pyparsing as pyp
    
pyp.ParserElement.setDefaultWhitespaceChars("\r\n\t &;")
pyp.ParserElement.enablePackrat()

class Grammar:
    def _re_flags(self,ignorecase):
        if ignorecase:
            return re.IGNORECASE
        else:
            return 0

    def _literal_cls(self,ignorecase):
        if ignorecase:
            return pyp.CaselessLiteral
        else:
            return pyp.Literal

    def _init_default_values(self):
        self.CLAUSE_NOT_FOUND = -2
        self.CLAUSE_VALUE_NOT_SPECIFIED = -1

    def _init_basic_tokens(self):
        self.LPAR = pyp.Suppress("(")
        self.RPAR = pyp.Suppress(")")
        self.EQ = pyp.Suppress("=")
        self.ELEM = pyp.Suppress("%")
        self.COMMA = pyp.Suppress(",")
        self.DOT = pyp.Suppress(".")
        self.UNDERSCORE = pyp.Suppress("_")
        self.PEQ = pyp.Suppress("=>")
        self.COLONS = pyp.Suppress("::")
        self.OPTCOMMA = pyp.Optional(self.COMMA)
        self.MATLPAR = pyp.Regex(r"\(\/|\[").suppress()
        self.MATRPAR = pyp.Regex(r"\/\)|\]").suppress()
        self.PRAGMA = pyp.Regex(r"[!c\*]\$").suppress()
  
    def _init_keywords(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        self.CALL = literal_cls("self.CALL")
        self.ATTRIBUTES = literal_cls("ATTRIBUTES")
        self.LEN = literal_cls("LEN")
        self.KIND = literal_cls("self.KIND")
        self.DIM = literal_cls("DIM")
        self.REAL = literal_cls("REAL")
        self.FLOAT = literal_cls("FLOAT")
        self.DBLE = literal_cls("DBLE")
        self.CMPLX = literal_cls("CMPLX")
        self.DCMPLX = literal_cls("DCMPLX")
        self.AIMAG = literal_cls("AIMAG")
        self.CONJG = literal_cls("CONJG")
        self.DCONJG = literal_cls("DCONJG")
        self.SIZE = literal_cls("SIZE")
        self.LBOUND = literal_cls("LBOUND")
        self.UBOUND = literal_cls("UBOUND")
        self.WHILE = literal_cls("self.WHILE")
        self.IF = literal_cls("IF")
        self.ELSE = literal_cls("ELSE")
        self.THEN = literal_cls("THEN")
        self.DO = literal_cls("DO")
        self.END = literal_cls("END")
        self.SELECT = literal_cls("SELECT")
        self.CASE = literal_cls("CASE")
        self.DEFAULT = literal_cls("DEFAULT")
        self.WHERE = literal_cls("WHERE")
        self.FORALL = literal_cls("FORALL")
        self.FOREACH = literal_cls("FOREACH")

    def _init_data_types(self,ignorecase):
        flags = self._re_flags(ignorecase)
        self.identifier = pyp.pyparsing_common.identifier.copy()
        self.identifier_no_action = self.identifier.copy()
        self.number = pyp.Regex(r"[+-]?(\.\d+|\d+(\.\d*)?)([eEdD]([+-]?\d+(\.\d*)?))?(_\w+)?")
        self.logical = pyp.Regex(r"\.(true|false)\.?(_\w+)?", flags)
        self.character = pyp.QuotedString("'", escQuote="\\")

    def _init_arith_expr(self,
          ignorecase,
          unary_op_parse_action,
          binary_op_parse_action):
        flags = self._re_flags(ignorecase)
        precedence_ordered_op_list = [
          (pyp.Regex(r"\.(?!\b(false|true|[gl][te]|eq|ne|not|and|or|xor|eqv|neqv)\b)[a-zA-Z]+\.",flags),
            1,pyp.opAssoc.RIGHT), # custom unary op (negative lookahead excludes taken tokens)
          (pyp.Regex(r"\*\*"), 2, pyp.opAssoc.RIGHT), # weirdly enough, -2**2**3 in Fortran is -pow(2,pow(2,3)) in C
          (pyp.Regex(r"[*/]"), 2, pyp.opAssoc.LEFT),
          (pyp.Regex(r"[+-]"), 1, pyp.opAssoc.RIGHT),
          (pyp.Regex(r"[+-]"), 2, pyp.opAssoc.LEFT),
          (pyp.Regex(r"<=?|=?>|[/=]=|\.(eq|ne|[gl][te])\.",flags), 
            2, pyp.opAssoc.LEFT),
          (pyp.Regex(r"\.not\.",flags),1,pyp.opAssoc.RIGHT),
          (pyp.Regex(r"\.and\.",flags),2,pyp.opAssoc.LEFT),
          (pyp.Regex(r"\.or\.",flags),2,pyp.opAssoc.LEFT),
          (pyp.Regex(r"\.\(xor\|eqv\|neqv\)\.",flags), 
            2, pyp.opAssoc.LEFT),
          (pyp.Regex(r"\.(?!\b(false|true|[gl][te]|eq|ne|not|and|or|xor|eqv|neqv)\b)[a-zA-Z]+\.",flags),
            2,pyp.opAssoc.LEFT), # custom binary op (negative lookahead excludes taken tokens)
          #(pyp.Regex(r"=",flags), 
          # 2, pyp.opAssoc.RIGHT),
        ]
     
        arith_expr_op_list = []
        for tup in precedence_ordered_op_list:
            expr, num_ops, opassoc = tup
            if num_ops == 1:
                if unary_op_parse_action != None:
                    arith_expr_op_list.append(
                      (expr, num_ops, opassoc, unary_op_parse_action)
                    )
                else:
                    arith_expr_op_list.append(tup)
            if num_ops == 2:
                if binary_op_parse_action != None:
                    arith_expr_op_list.append(
                      (expr, num_ops, opassoc, binary_op_parse_action)
                    )
                else:
                    arith_expr_op_list.append(tup)
        self.arith_expr <<= pyp.infixNotation( # note: forward declared
          ( self.rvalue | self.complex_arith_expr ),
          arith_expr_op_list
        )

    def _init_arithmetic_expressions(self,
          ignorecase,
          unary_op_parse_action,
          binary_op_parse_action
        ):
        # arithmetic logical expressions and assignments
        self.complex_arith_expr = pyp.Forward()
        self.arith_expr = pyp.Forward()
        
        self.function_call_arg = pyp.Forward()
        self.function_call_args = pyp.Optional(pyp.delimitedList(self.function_call_arg))
        self.function_call = ( 
          self.identifier_no_action 
          + self.LPAR
          + self.function_call_args
          + self.RPAR # emits 1* tokens
        )
        
        # derived_types
        derived_type_rvalue = pyp.Forward()
        self.derived_type_elem = (
          ( self.function_call | self.identifier ) 
          + self.ELEM 
          + derived_type_rvalue 
        )
        derived_type_rvalue <<= self.derived_type_elem | self.function_call | self.identifier
       
        # self.rvalue, self.lvalue 
        #self.conversion = pyp.Forward()
        #self.inquiry_function = pyp.Forward()
        self.rvalue = (
          self.derived_type_elem 
          | self.function_call 
          | self.identifier 
          | self.logical 
          | self.character 
          | self.number
        )# |: ordered OR, order is import
        # legacy
        #self.rvalue = (
        #  self.conversion 
        #  | self.inquiry_function 
        #  | self.derived_type_elem 
        #  | self.function_call 
        #  | self.identifier 
        #  | self.logical 
        #  | self.character 
        #  | self.number
        #)# |: ordered OR, order is import
        self.lvalue = (
          self.derived_type_elem 
          | self.function_call 
          | self.identifier
        )
       
        self._init_arith_expr(
          ignorecase,
          unary_op_parse_action,
          binary_op_parse_action
        )
        
        self.assignment_begin = self.lvalue + self.EQ
        self.assignment = self.lvalue + self.EQ + self.arith_expr # ! emits 2 tokens: *,*
        self.keyword_argument = self.identifier_no_action + self.EQ + self.arith_expr
        
        self.matrix_arith_expr = self.MATLPAR + pyp.Group(pyp.delimitedList(self.arith_expr)) + self.MATRPAR
        self.matrix_assignment = self.lvalue + self.EQ + self.matrix_arith_expr
        
        self.complex_arith_expr <<= self.LPAR + pyp.Group(self.arith_expr + self.COMMA + self.arith_expr) + self.RPAR 
        self.complex_assignment = self.lvalue + self.EQ + self.complex_arith_expr
        
        self.fortran_assignment = ( 
          self.matrix_assignment
          | self.complex_assignment
          | self.assignment
        )
        COLON = pyp.Literal(":").suppress()
        opt_arith_expr = pyp.Optional(self.arith_expr,default=None)
        self.tensor_slice = ( 
          ( opt_arith_expr + COLON + opt_arith_expr + COLON + self.arith_expr )
          | ( opt_arith_expr + COLON + opt_arith_expr )
        )
        # define forward declared tokens
        self.function_call_arg <<= self.tensor_slice | self.keyword_argument | self.arith_expr
        
        # conversion functions
        # todo: check type of variable when translating
        # see http://userweb.eng.gla.ac.uk/peter.smart/com/com/f77-conv.htm
        # for now, always assume conversion between complex and float types
        # todo: handle directly in function call
        # todo: better support for intrinsics in indexer
        #func_kind = (
        #  self.COMMA 
        #  + ( self.KIND + self.EQ + self.arith_expr )
        #) # emits one token # todo: generalize to keyword argument
        #single_arg_plus_kind = (
        #  self.LPAR 
        #  + (~func_kind+self.arith_expr) 
        #  + pyp.Optional(func_kind,default=None) 
        #  + self.RPAR
        #) # emits 2 tokens: *,*,*
        #double_arg_plus_kind = (
        #  self.LPAR 
        #  + (~func_kind+self.arith_expr) 
        #  + pyp.Optional(self.COMMA 
        #  + (~func_kind+self.arith_expr),default="0") 
        #  + pyp.Optional(func_kind,default=None)
        #  + self.RPAR 
        #) # emits 2 tokens: *,*
        #
        #self.convert_to_extract_real   = ( self.REAL | self.FLOAT ) + single_arg_plus_kind # emits 2 tokens,
        #self.convert_to_double         = self.DBLE   + single_arg_plus_kind # emits 2 tokens,
        #self.convert_to_complex        = self.CMPLX  + double_arg_plus_kind # emits 3 tokens, op (x,y) -> x+iy | c.x = x, c.y = b  ; op: x -> (x,0) -> x+i0 -> | c.x = x, c.y =0
        #self.convert_to_double_complex = self.DCMPLX + double_arg_plus_kind # emits 3 tokens, op (x,y) -> x+iy | c.x = x, c.y = b  ; op: x -> (x,0) -> x+i0 -> | c.x = x, c.y =0
        #self.extract_imag              = self.AIMAG  + single_arg_plus_kind # emits 1 tokens, op: x+iy -> y
        #self.conjugate                 = self.CONJG  + single_arg_plus_kind # emits 1 tokens, op: x+iy -> x-iy | c.y = -c.y
        #self.conjugate_double_complex  = self.DCONJG + single_arg_plus_kind # emits 1 tokens, op: x+iy -> x-iy | c.y = -c.y
        #
        #self.conversion <<= (
        #  self.convert_to_extract_real
        #  | self.convert_to_double
        #  | self.convert_to_complex
        #  | self.convert_to_double_complex
        #  | self.extract_imag
        #  | self.conjugate
        #  | self.conjugate_double_complex
        #)        
        ## inquiry functions
        #inquiry_function_arg = ( 
        #  self.lvalue 
        #  + pyp.Optional(
        #      self.COMMA
        #      + pyp.Optional(self.DIM + self.EQ) 
        #      + self.arith_expr,default=None
        #  ) 
        #  + pyp.Optional(
        #      self.COMMA
        #      + pyp.Optional(self.KIND + self.EQ)
        #      + self.arith_expr,default=None
        #  )
        #)
        #self.size_inquiry = self.SIZE   + self.LPAR + inquiry_function_arg + self.RPAR  
        #self.lbound_inquiry = self.LBOUND + self.LPAR + inquiry_function_arg + self.RPAR 
        #self.ubound_inquiry = self.UBOUND + self.LPAR + inquiry_function_arg + self.RPAR 
        #self.inquiry_function <<= (
        #  self.size_inquiry
        #  | self.lbound_inquiry
        #  | self.ubound_inquiry
        #)

    def _init_fortran_statements(self,ignorecase):
        self.fortran_subroutine_call = self.CALL + self.function_call

    def _init_cuda_fortran_keywords(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        self.LCHEVRON = pyp.Suppress(">>>")
        self.RCHEVRON = pyp.Suppress(">>>")
        self.CUBLAS = literal_cls("cublas")
        self.ALLOCATE = pyp.Regex(r"\ballocate\b",flags).suppress()
        self.ALLOCATED = pyp.Regex(r"\ballocatedb",flags).suppress()
        self.DEALLOCATE = pyp.Regex(r"\bdeallocate\b",flags).suppress()
        self.CUDAMALLOC = literal_cls("cudamalloc")
        self.CUDAMEMCPY = literal_cls("cudamemcpy")
        self.CUDAMEMCPYASYNC = literal_cls("cudamemcpyasync")
        self.CUDAMEMCPY2D = literal_cls("cudamemcpy2d")
        self.CUDAMEMCPY2DASYNC = literal_cls("cudamemcpy2dasync")
        self.CUDAMEMCPY3D = literal_cls("cudamemcpy3d")
        self.CUDAMEMCPY3DASYNC = literal_cls("cudamemcpy3dasync")

    def _init_cuf_kernel_do_expressions(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        #
        self.cuf_kernel_do_auto = pyp.Literal("*")
        self.cuf_kernel_do_dim3 = (
          self.LPAR 
          + pyp.delimitedList(
              self.arith_expr 
              | self.cuf_kernel_do_auto
            )
          + self.RPAR
        )
        
        self.cuf_kernel_do_arg_grid  = (
          self.cuf_kernel_do_auto 
          | self.cuf_kernel_do_dim3 
          | self.rvalue
        )
        self.cuf_kernel_do_arg_block = self.cuf_kernel_do_arg_grid.copy()
        self.cuf_kernel_do_arg_sharedmem = pyp.Group(self.arith_expr) # group because single expression
        self.cuf_kernel_do_arg_stream = pyp.Group(self.arith_expr)
        
        self.cuf_kernel_do_launch_params_ext1 = (
          self.cuf_kernel_do_arg_sharedmem + self.COMMA
          + self.cuf_kernel_do_arg_stream
        )
        
        self.cuf_kernel_do_launch_params_ext2 = (
          pyp.Regex(r"stream\s*=",flags).suppress() 
          + self.cuf_kernel_do_arg_stream
        )
        
        self.cuf_kernel_do_launch_params = pyp.Group(
          self.LCHEVRON
          + self.cuf_kernel_do_arg_grid  + self.COMMA 
          + self.cuf_kernel_do_arg_block
          + ( 
            self.COMMA + self.cuf_kernel_do_launch_params_ext2 + self.RCHEVRON
            | self.COMMA + self.cuf_kernel_do_launch_params_ext1 + self.RCHEVRON
            | self.RCHEVRON
          )
        )
        self.cuf_kernel_do_arg_num_loops = pyp.Optional(
          self.LPAR + self.arith_expr + self.RPAR,
          default=None
        )
        
        # directives
        self.cuf_kernel_do =(
          self.PRAGMA + pyp.Regex(r"cuf\s*kernel\s*do*",flags).suppress() 
          + self.cuf_kernel_do_arg_num_loops 
          + pyp.Optional(self.cuf_kernel_do_launch_params,default=None)
        )

    def _init_cuda_fortran_expressions(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        #
        self._init_cuda_fortran_keywords(ignorecase)
        self._init_cuf_kernel_do_expressions(ignorecase)
        
        # scanner/analysis
        self.allocated = self.ALLOCATED  + self.LPAR + self.rvalue + self.RPAR
        self.memcpy = self.rvalue + self.EQ + self.rvalue + ( pyp.Suppress(";") | pyp.LineEnd() )
        self.non_zero_check = self.rvalue + pyp.Regex(r"/|=|\.ne\.",flags).suppress() + pyp.Suppress("0")
        self.pointer_assignment = self.rvalue + self.PEQ + self.rvalue
        
        stream_arg = self.cuf_kernel_do_arg_stream
        
        # scanner/analysis
        # dest,count # kind is inferred from dest and src
        self.cuf_cudamalloc = (
          self.CUDAMALLOC 
          + self.LPAR 
          + self.identifier 
          + self.COMMA 
          + self.arith_expr 
          + self.RPAR
        )
        # dest,src,count,[,stream] # kind is inferred from dest and src
        cuda_memcpy_type = pyp.Regex(
          r"\b(cudamemcpyhosttohost|cudamemcpyhosttodevice|cudamemcpydevicetohost|cudamemcpydevicetodevice)\b",
          flags
        )
        cuda_memcpy_args_ext  = (
          pyp.Optional(self.COMMA + cuda_memcpy_type,default=None)
          + pyp.Optional(self.COMMA + stream_arg,default=None)
        )
        cuf_cudamemcpy_args  = (
          self.rvalue
          + self.COMMA
          + self.rvalue
          + self.COMMA
          + self.arith_expr 
          + cuda_memcpy_args_ext
        )
        self.cuf_cudamemcpy = (
          ( self.CUDAMEMCPYASYNC | self.CUDAMEMCPY )
          + self.LPAR
          + cuf_cudamemcpy_args
          + self.RPAR
        )
        # dest,dpitch(count),src,spitch(count),width(count),height(count)[,stream] # kind is inferred from dest and src
        cuf_cudamemcpy2d_args = (
          self.rvalue
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.rvalue
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.arith_expr
          + cuda_memcpy_args_ext
        )
        self.cuf_cudamemcpy2d = (
          ( self.CUDAMEMCPY2D | self.CUDAMEMCPY2DASYNC )
          + self.LPAR
          + cuf_cudamemcpy2d_args
          + self.RPAR
        )
        # dest,dpitch(count),src,spitch(count),width(count),height(count),depth(count),[,stream] # kind is inferred from dest and src
        cuf_cudamemcpy3d_args = (
          self.rvalue
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.rvalue
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.arith_expr
          + self.COMMA
          + self.arith_expr 
          + cuda_memcpy_args_ext
        )
        self.cuf_cudamemcpy3d = (
          ( self.CUDAMEMCPY3D | self.CUDAMEMCPY3DASYNC ) 
          + self.LPAR 
          + cuf_cudamemcpy3d_args 
          + self.RPAR
        )
        self.cuf_cudamemcpy_variant = (
          self.cuf_cudamemcpy
          | self.cuf_cudamemcpy2d
          | self.cuf_cudamemcpy3d
        )
        # cublas/analysis
        self.CUBLAS_OPERATION_TYPE = pyp.Regex(r"'[ntc]'",flags)#.setParseAction(lambda tokens: "hipblas_op_"+tokens[0].strip("'").upper())
        #cublas_arglist       = pyp.Group(pyp.delimitedList(cublas_operation_type | self.rvalue))
        # todo: Explicitly scan for ttrvalues in cublas_arglist's self.arith_expr when transforming host code
        cublas_arglist = pyp.Group(
          pyp.delimitedList(self.CUBLAS_OPERATION_TYPE | self.arith_expr)
        ) 
        self.cuf_cublas_call = (
          self.CUBLAS.suppress()
          + self.identifier
          + self.LPAR
          + cublas_arglist
          + self.RPAR
        )  # emits 2 tokens
        
        # anchors; todo: simplify 
        cuda_api = pyp.Combine(
          pyp.Regex(r"\b(cublas|cufft|cusparse|cuda|cusolver)",flags) 
          + self.identifier
        )
        # cuda_lib_call is used to detect any CUDA library calls;
        # they are then analysed and transformed using more specific constructs
        self.cuda_lib_call = (
          ((self.identifier + self.EQ) | self.CALL).suppress()
          + cuda_api 
          + self.LPAR 
          + pyp.Optional(self.function_call_args,default=None) 
          + self.RPAR 
        ) # emits 3 tokens -> *,

    def _init_gpufort_control_directives(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        #
        self.gpufort_control = literal_cls("!$gpufort").suppress() + pyp.Regex(r"on|off",flags)

    def _init_openacc_keywords(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        #
        self.ACC = literal_cls("acc").suppress()
        self.KERNELS = literal_cls("kernels").suppress()
        self.PARALLEL = literal_cls("parallel").suppress()
        self.LOOP = literal_cls("loop").suppress()
        self.DATA = literal_cls("data").suppress()
        self.ENTER = literal_cls("enter").suppress()
        self.EXIT = literal_cls("exit").suppress()
        self.HOST_DATA = literal_cls("host_data").suppress()
        self.ATOMIC = literal_cls("atomic").suppress()
        self.UPDATE = literal_cls("update").suppress()
        self.SERIAL = literal_cls("serial").suppress()
        self.CACHE = literal_cls("cache").suppress()
        self.ROUTINE = literal_cls("routine").suppress()
        self.DECLARE = literal_cls("declare").suppress()
        self.ASYNC = literal_cls("async").suppress()
        self.WAIT = literal_cls("wait").suppress()
        self.NUM_GANGS = literal_cls("num_gangs").suppress()
        self.NUM_WORKERS = literal_cls("num_workers").suppress()
        self.VECTOR_LENGTH = literal_cls("vector_length").suppress()
        self.DEVICE_TYPE = literal_cls("device_type").suppress()
        self.COPY = literal_cls("copy").suppress()
        self.COPYIN = literal_cls("copyin").suppress()
        self.COPYOUT = literal_cls("copyout").suppress()
        self.CREATE = literal_cls("create").suppress()
        self.NO_CREATE = literal_cls("no_create").suppress()
        self.PRESENT = literal_cls("present").suppress()
        self.DEVICEPTR = literal_cls("deviceptr").suppress()
        self.ATTACH = literal_cls("attach").suppress()
        self.DETACH = literal_cls("detach").suppress()
        self.DEFAULT = literal_cls("default").suppress()
        self.NONE = literal_cls("none").suppress()
        self.PRIVATE = literal_cls("private").suppress()
        self.FIRST_PRIVATE = literal_cls("first_private").suppress()
        self.REDUCTION = literal_cls("reduction").suppress()
        self.DELETE = literal_cls("delete").suppress()
        self.ROUTINE = literal_cls("routine").suppress()
        self.USE_DEVICE = literal_cls("use_device").suppress()
        self.COLLAPSE = literal_cls("collapse").suppress()
        self.SELF = literal_cls("self").suppress()
        self.DEVICE = literal_cls("device").suppress()
        self.HOST = literal_cls("host").suppress()
        self.BIND = literal_cls("bind").suppress()
        self.DEVICE_RESIDENT = literal_cls("device_resident").suppress()
        self.LINK = literal_cls("link").suppress()
        self.TILE = literal_cls("tile").suppress()
        self.GANG = literal_cls("gang").suppress()
        self.VECTOR = literal_cls("vector").suppress()
        self.WORKER = literal_cls("worker").suppress() 
        self.ACC_REDUCTION_OP = pyp.Regex(r"\+|\*|\b(max|min|iand|ior|ieor)\b|\.(and|or|eqv|neqv)\.",flags)
        self.DEVICE_TYPE = pyp.Regex(r"\b(dtype|device_type)\b",flags).suppress()
        self.acc_noarg_clause = pyp.Regex(r"\b(seq|auto|independent|read|write|capture|nohost|finalize|if_present)\b",flags)
        self.acc_routine_noarg_clause = pyp.Regex(r"\b(seq|auto|independent|read|write|capture|nohost|finalize|if_present)\b",flags)

    def _init_openacc_directives(self,ignorecase):
        literal_cls = self._literal_cls(ignorecase)
        flags = self._re_flags(ignorecase)
        #
        self.acc_var_list = self.LPAR + pyp.Group(pyp.delimitedList(self.rvalue)) + self.RPAR
        
        # clauses
        self.acc_clause_gang = self.GANG + pyp.Optional(self.LPAR + self.arith_expr + self.RPAR,default = None)
        self.acc_clause_worker = self.WORKER + pyp.Optional(self.LPAR + self.arith_expr + self.RPAR,default = None)
        self.acc_clause_vector = self.VECTOR + pyp.Optional(self.LPAR + self.arith_expr + self.RPAR,default = None)
        self.acc_clause_num_gangs = self.NUM_GANGS + self.LPAR + self.arith_expr + self.RPAR 
        self.acc_clause_num_workers = self.NUM_WORKERS + self.LPAR + self.arith_expr + self.RPAR 
        self.acc_clause_vector_length = self.VECTOR_LENGTH + self.LPAR + self.arith_expr + self.RPAR
        
        self.acc_clause_device_type = self.DEVICE_TYPE + self.LPAR + pyp.Group(pyp.delimitedList(self.identifier) | pyp.Literal("*")) + self.RPAR
        self.acc_clause_if = self.IF + self.LPAR + self.arith_expr + self.RPAR
        
        # copy, copyin, copyout, create, no_create, present, deviceptr, attach, detach, use_device, delete, private, first_private, host, device_resident, link
        self.acc_mapping_clause = self.identifier.copy() + self.acc_var_list
        
        self.acc_clause_default = self.DEFAULT + self.LPAR + pyp.Regex(r"none|present",flags) + self.RPAR # do not suppress
        self.acc_clause_reduction = self.REDUCTION + self.LPAR + self.ACC_REDUCTION_OP + pyp.Suppress(":") + pyp.Group(pyp.delimitedList(self.rvalue)) + self.RPAR
        self.acc_clause_collapse = self.COLLAPSE + self.LPAR + self.arith_expr + self.RPAR
        self.acc_clause_self = self.SELF + self.LPAR + self.arith_expr + self.RPAR # for compute constructs; not for update
        self.acc_clause_bind = self.BIND + self.LPAR + self.identifier + self.RPAR
        self.acc_clause_tile = self.TILE + self.LPAR + pyp.Group(pyp.delimitedList(self.arith_expr)) + self.RPAR
        self.acc_clause_wait = self.WAIT + pyp.Optional(pyp.Group(pyp.delimitedList(self.arith_expr)), default = [])
        self.acc_clause_async = self.ASYNC + pyp.Optional(self.LPAR + self.rvalue + self.RPAR, default = self.CLAUSE_VALUE_NOT_SPECIFIED) 
        
        # self.ACC LoopNest directive
        self.ACC_START = self.PRAGMA + self.ACC 
        self.ACC_END = self.ACC_START.copy() + self.END
        
        self.acc_clause = (
          self.acc_clause_if
          | self.acc_clause_self
          | self.acc_clause_async
          | self.acc_clause_wait
          | self.acc_clause_num_gangs
          | self.acc_clause_num_workers
          | self.acc_clause_vector_length
          | self.acc_clause_device_type
          | self.acc_clause_default
          | self.acc_clause_reduction
          | self.acc_clause_collapse
          | self.acc_clause_bind
          | self.acc_clause_tile
          | self.acc_clause_gang
          | self.acc_clause_worker
          | self.acc_clause_vector
          | self.acc_mapping_clause
          | self.acc_noarg_clause
        )
       
        # todo: can be simplified as multi-line statementsa re collapsed
        self.acc_clause_list = pyp.Group(
          pyp.Optional(
            pyp.OneOrMore(pyp.Optional(self.ACC_START + pyp.Literal("&")).suppress() + self.acc_clause),
            default=[]
          )
        )
        
        self.ACC_END_HOST_DATA = self.ACC_END + self.HOST_DATA
        self.ACC_END_DATA = self.ACC_END + self.DATA
        self.ACC_END_ATOMIC = self.ACC_END + self.ATOMIC
        
        self.acc_update = self.ACC_START + self.UPDATE + self.acc_clause_list # 2 tokens -> [(False|True),[*]],*
        self.acc_wait = self.ACC_START + self.acc_clause_wait + self.acc_clause_list
        self.acc_host_data = self.ACC_START + self.HOST_DATA + self.acc_clause_list
        self.acc_data = self.ACC_START + self.DATA + self.acc_clause_list
        self.acc_enter_data = self.ACC_START + self.ENTER + self.DATA + self.acc_clause_list
        self.acc_exit_data = self.ACC_START + self.EXIT  + self.DATA + self.acc_clause_list
        self.acc_routine = ( 
          self.ACC_START 
          + self.ROUTINE 
          + pyp.Optional(
            self.LPAR 
            + self.identifier 
            + self.RPAR,
            default = None
          ) 
          + self.acc_clause_list
        )
        self.acc_declare = self.ACC_START + self.DECLARE + self.acc_clause_list
        self.acc_atomic = self.ACC_START + self.ATOMIC + self.acc_clause_list
        self.acc_cache = self.ACC_START + self.CACHE + self.LPAR + self.acc_var_list + self.RPAR
        self.acc_loop = self.ACC_START + self.LOOP + self.acc_clause_list
        
        # kernels / parallels
        self.acc_serial = self.ACC_START + self.SERIAL   + self.acc_clause_list
        self.acc_kernels = self.ACC_START + self.KERNELS  + ~self.LOOP + self.acc_clause_list
        self.acc_parallel = self.ACC_START + self.PARALLEL + ~self.LOOP + self.acc_clause_list
        self.acc_parallel_loop = self.ACC_START + self.PARALLEL + self.LOOP  + self.acc_clause_list
        self.acc_kernels_loop = self.ACC_START + self.KERNELS  + self.LOOP  + self.acc_clause_list
        
        self.ACC_END_SERIAL = self.ACC_END + self.SERIAL
        self.ACC_END_KERNELS = self.ACC_END + self.KERNELS  + ~self.LOOP
        self.ACC_END_PARALLEL = self.ACC_END + self.PARALLEL + ~self.LOOP
        self.ACC_END_PARALLEL_LOOP = self.ACC_END + self.PARALLEL + self.LOOP # copy() allows us to attach different parse action
        self.ACC_END_KERNELS_LOOP = self.ACC_END + self.KERNELS  + self.LOOP
        
        # manual grouping
        self.acc_simple_directive = (
            self.acc_host_data 
          | self.acc_data 
          | self.acc_enter_data 
          | self.acc_exit_data 
          | self.acc_loop 
          | self.acc_routine 
          | self.acc_declare 
          | self.ACC_END_DATA
        ) # self.TBC
        
        self.acc_host_directive = (
            self.acc_kernels_loop 
          | self.acc_parallel_loop 
          | self.acc_kernels 
          | self.acc_parallel 
          | self.acc_simple_directive 
          | self.ACC_END_DATA 
          | self.ACC_END_HOST_DATA
        )

    def __init__(self,
                 ignorecase=False,
                 unary_op_parse_action=None,
                 binary_op_parse_action=None):
        """
        :param bool ignorecase: Create case-insensitive parser for Fortran keywords.
                                 Otherwise, assume lower case.
        :param unary_op_parse_action: Pyparsing parse action (function or class) to run (or init) when parsing unary operator applications.
                                      Look up pyparsing parse actions for more details.
        :param binary_op_parse_action: Pyparsing parse action (function or class) to run (or init) when parsing binary operator applications.
                                      Look up pyparsing parse actions for more details.
        """
        # basic
        self._init_default_values()
        self._init_basic_tokens()
        self._init_keywords(ignorecase)
        self._init_data_types(ignorecase)
        self._init_arithmetic_expressions(
          ignorecase,
          unary_op_parse_action,
          binary_op_parse_action
        ) 
        self._init_fortran_statements(ignorecase) 
        self._init_gpufort_control_directives(ignorecase)
        #
        self._init_cuda_fortran_expressions(ignorecase)
        #
        self._init_openacc_keywords(ignorecase)
        self._init_openacc_directives(ignorecase)

    # API
    def parse_arith_expr(self,expr,parse_all=True):
        return self.arith_expr.parseString(
          expr,parseAll=parse_all
        )[0]
    
    def parse_assignment(self,expr,parse_all=True):
        return self.assignment.parseString(
          expr,parseAll=parse_all
        )[0]
