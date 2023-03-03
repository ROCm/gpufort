# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import enum
import copy

from gpufort import util    
from gpufort import translator

statement_classifier = util.parsing.StatementClassifier()

class ValueType(enum.Enum):
    UNKNOWN = 0
    PROCEDURE = 1
    VARIABLE = 2
    INTRINSIC = 3

DEFAULT_IMPLICIT_SPEC =\
  statement_classifier.parse_implicit_statement(
    "IMPLICIT integer (i-n), real (a-h,o-z)")

EMPTY_TYPE = {
   "name": None,
   "kind": None,
   "attributes": [],
   "accessibility": None, # default accessibility of all vars
   "public": [], 
   "private": [], 
   "params": [],
   "variables": [],
   # meta information
   "file": None,
   "lineno": None,
   # dummy entries that never carry data,
   # they make the type look like a module to certain routines
   # todo: should use dict.get(key,default) instead.
   "types": [],
}

EMPTY_PROCEDURE = {
  "name": None,
  "kind": None,
  "result_name": None,
  "attributes": [],
  "dummy_args": [],
  "variables": [],
  "types": [],
  "procedures": [],
  "used_modules": [],
  # meta information
  "file" : None,
  "lineno" : -1,
}

EMPTY_SCOPE = {"tag": "", "types": [], "variables": [], "procedures": [], "index": [], "implicit": None}
SCOPE_ENTRY_TYPES = ["types", "variables", "procedures"]

EMPTY_VAR = {
  "name"   : None,
  "f_type" : None,
  "len"    : None,
  "kind"   : None,
  "params" : [],
  # todo: bytes per element can be computed on the fly
  "bytes_per_element" : None,
  "c_type" : None,
  "attributes" : [],
  # ACC/OMP
  "declare_on_target" : None,
  # arrays
  "bounds" : [],
  "rank"   : -1,
  # parse rhs if necessary
  "rhs" : None,
  # meta information
  "module": None, # todo: Compare vs parent_tag in scope variables
  "file" : None,
  "lineno" : -1,
}

def new_scope():
    return copy.deepcopy(EMPTY_SCOPE)

def copy_scope(existing_scope,index=None,tag=None,implicit=None):
    """
    :note: All scope entry lists but not their members (dict references)
           are copied as it is assumed that the copied scope's lists
           are modified later on.
           Scope entry list members can be numerous and large,
           so a simple deep copy can quickly become too time consuming.
    :note: Implicit rule is local to a scope and hence set to None here.
    :note: 
    """
    copied_scope = new_scope()
    for scope_entry in SCOPE_ENTRY_TYPES:
        copied_scope[scope_entry] += existing_scope[scope_entry] 
    copied_scope["tag" ] = (
      existing_scope["tag"] if tag == None
      else tag
    )
    copied_scope["index" ] = (
      existing_scope["index"] if index == None 
      else index
    )
    copied_scope["implicit" ] = (
      existing_scope["implicit"] if implicit == None
      else implicit
    )
    return copied_scope

def create_index_var(f_type,
                     f_len,
                     kind,
                     params,
                     name,
                     attributes=[],
                     bounds=[],
                     rhs=None,
                     module=None,
                     filepath="<unknown>",
                     lineno=-1):
    ivar = copy.deepcopy(EMPTY_VAR)
    # basic
    ivar["name"]        = name
    ivar["f_type"]      = f_type
    ivar["kind"]        = kind
    ivar["len"]         = f_len
    ivar["params"]      = params
    # todo: bytes per element can be computed on the fly
    ivar["attributes"] += attributes
    # arrays
    ivar["bounds"] += bounds
    ivar["rank"]   = len(bounds)
    # handle parameters
    #ivar["value"] = None # todo: parse rhs if necessary
    ivar["rhs"] = rhs
    # meta information
    ivar["file"] = filepath
    ivar["lineno"] = lineno
    return ivar

def render_datatype(ivar):
    datatype = ivar["f_type"]
    args = []
    if datatype == "character":
        if ivar["len"] != None:
            args.append(ivar["len"])
    if datatype != "type" and ivar["kind"] != None:
        args.append(ivar["kind"])
    elif datatype == "type":
        arg1 = ivar["kind"]
        if len(ivar["params"]):
            arg1.append("(")
            arg1.append(",".join(ivar["params"]))
            arg1.append(")")
        args.append("".join(arg1))
    if len(args):
        return datatype + "(" + ",".join(args) + ")"
    else:
        return datatype
    
def render_declaration(ivar):
    result = [render_datatype(ivar)]
    if len(ivar["attributes"]):
        result += [", ",", ".join(ivar["attributes"])]
    result += [" :: ",ivar["name"]]
    if len(ivar["bounds"]):
        result += ["(",", ".join(ivar["bounds"]),")"]
    if ivar["rhs"] != None:
        if "pointer" in ivar["attributes"]:
            result += [" => ",ivar["rhs"]]
        else:
            result += [" = ",ivar["rhs"]]
    return "".join(result)

class IndexRecordAccessorBase:
    @property
    def name(self):
        return self.record["name"]
    @property
    def kind(self):
        return self.record["kind"]
    @property
    def file(self):
        return self.record["file"]
    @property
    def lineno(self):
        return self.record["lineno"]
    def __getitem__(self,key):
        """:deprecated: Rather use the explicitly defined properties."""
        return self.record[key] 
    def __setitem__(self, key, value):
        """:deprecated:"""
        self.record[key] = value
    def __eq__(self, other):
        if isinstance(other,IndexRecordAccessorBase):
            return self.record == other.record
        else:
            return False
    def __hash__(self, other):
        return hash(self.record)

class IndexFortranConstructBase(IndexRecordAccessorBase):
    @property
    def variables(self):
        for ivar in self.record["variables"]:
            yield IndexVariable(ivar,ivar["name"])
    def get_variable(self,name):
        name = name.lower()
        for ivar in self.variables:
            if ivar.name == name:
                return ivar
        raise util.error.LookupError(
          "no index record found for variable"
          + "'{}' of procedure '{}'".format(
            name,
            self.record["name"]
          )
        )
    @property
    def procedures(self):
        """:return: List of nested procedures as IndexProcedure."""
        for iproc in self.record["procedures"]:
            yield IndexProcedure(iproc)
    @property
    def types(self):
        """:return: List of nested types as IndexType."""
        for itype in self.record["types"]:
            yield IndexType(itype)
    @property
    def used_modules(self):
        return self.record["used_modules"]        
    @property
    def implicit(self):
        return self.record["implicit"]
    @property
    def accessibility(self):
        """:return: Default accessibility of the type's members
                    ("public" or "private")."""
        return self.record["accessibility"]
    @property
    def _members_explicity_set_public(self):
        """:return: Names of explicitly named public members."""
        return self.record["public"]
    @property
    def _members_explicity_set_private(self):
        """:return: Names of explicitly named private members."""
        return self.record["private"]
    def is_private_member(self,member):
        """:return: If `member` is a private member."""
        if self.accessibility == "public":
            return member.lower() in self._members_explicity_set_private
        else:
            return member.lower() not in self._members_explicity_set_public
    def is_public_member(self,member):
        return not self.is_private(member_name)
    @property
    def public_variables(self):
        for var in self.variables:
            if self.is_public_member(var.name):
                yield var
    @property
    def private_variables(self):
        for var in self.variables:
            if self.is_private_member(var.name):
                yield var
    @property
    def public_procedures(self):
        for var in self.procedures:
            if self.is_public_member(var.name):
                yield var
    @property
    def private_procedures(self):
        for var in self.procedures:
            if self.is_private_member(var.name):
                yield var
    @property
    def public_types(self):
        for var in self.types:
            if self.is_public_member(var.name):
                yield var
    @property
    def private_types(self):
        for var in self.types:
            if self.is_private_member(var.name):
                yield var

class IndexModuleOrProgram(IndexFortranConstructBase):
    def __init__(self,record):
        self.record = record

class IndexType(IndexFortranConstructBase):
    def __init__(self,record):
        self.record = record
        #del self.types
        #del self.procedures
        #del self.public_procedures
        #del self.public_types
        #del self.private_procedures
        #del self.private_types
    @property
    def attributes(self):
        return self.record["attributes"]
    @property
    def is_c_interoperable(self):
        return "bind(c)" in self.attributes

class IndexProcedure(IndexFortranConstructBase):
    def __init__(self,record):
        self.record = record
        #del self.public_variables
        #del self.public_procedures
        #del self.public_types
        #del self.private_variables
        #del self.private_procedures
        #del self.private_types
        #del self.accessibility
        #del self._members_explicity_set_public
        #del self._members_explicity_set_private
        #del self.is_public_member
        #del self.is_private_member
    @property
    def is_subroutine(self):
        return self.record["kind"] == "subroutine"
    @property
    def is_function(self):
        return not self.is_subroutine
    @property
    def result_name(self):
        return self.record["result_name"]
    @property
    def result(self):
        assert self.is_function
        return self.get_variable(
          self.result_name
        )
    @property
    def name(self):
        return self.record["name"]
    @property
    def attributes(self):
        return self.record["attributes"]
    @property
    def is_intrinsic(self):
        return "intrinsic" in self.attributes
    @property
    def is_elemental(self):
        return "elemental" in self.attributes
    @property
    def is_conversion(self):
        return "conversion" in self.attributes
    @property
    def result_depends_on_kind(self):
        """:note: Conversions are always elemental."""
        return "kind_arg" in self.attributes
    @property
    def procedures(self):
        """:return: List of nested procedures as IndexProcedure."""
        for iproc in self.record["procedures"]:
            yield IndexProcedure(iproc)
    @property
    def types(self):
        """:return: List of nested types as IndexType."""
        for itype in self.record["types"]:
            yield IndexType(itype)
    @property
    def dummy_args(self):
        return self.record["dummy_args"]
    def get_argument(self,name):
        """:return: index variable for the given argument name.
        :param str name: Argument name, must be present in dummy arguments list.
        :see: IndexProcedure.dummy_args
        :raise util.error.LookupError: if no variable with the 
                                       given name could be found"""
        if name.lower() not in self.dummy_args:
            raise util.error.LookupError(
              "'{}' is no argument of procedure '{}'".format(
                name,
                self.name
              )
            )
        return self.get_variable(name)

class IndexVariable(IndexRecordAccessorBase):
    ANY_TYPE = "*"
    ANY_RANK = -2
    ANY_RANK_GREATER_ZERO = -1
  
    class Intent(enum.Enum):
        DEFAULT = 0
        IN = 1
        OUT = 2
        INOUT = 3

    def __init__(self,ivar,tag):
        self.record = ivar
        self._tag = tag
        #
        self._bytes_per_element = None
        self._resolved_kind = None
        self._resolved_rhs = None
        self._resolved_bounds = None
         
    @property
    def tag(self):
        """:return: A tag to better identify the variable, 
                    e.g., if it is a derived type member. 
        Example:

        If this would be derived type member a%b, the `name` field would 
        return `v` while this field would return `a%b`.

        :note: This field must be set by the routine / parent
        that instantiates this IndexVariable.
        """
        return self._tag
     
    @property
    def type(self):
        return self.record["f_type"]
    
    @property
    def kind(self):
        return self.record["kind"]
   
    @property
    def is_numeric(self):
        return self.type in ["integer","real","complex"]
    
    @property
    def is_logical(self):
        return self.type == "logical"
    
    @property
    def is_character(self):
        return self.type == "character"
    
    @property
    def is_derived_type(self):
        return self.type == "type"
 
    @property
    def len(self):
        """Length specifier of a character."""
        assert self.is_character
        return self.record["len"]
    
    @property
    def rhs(self):
        """:return: Right-hand side as string.
        """
        return self.record["rhs"]

    @property
    def matches_any_type(self):
        return self.type == "type" and self.kind == "*"

    def matches_type(self,typ):
        return (self.matches_any_type
                or self.type == typ)

    @property
    def bounds(self):
        return self.record["bounds"]
    @property
    def rank(self):
        """:return: Rank of the variable.
        :see: matches_rank
        """
        # todo: remove rank from index records
        bounds = self.bounds
        return len(bounds)

    @property
    def min_rank(self):
        """:return: Minimum accepted argument rank if an '*' or '..' is present in the array bounds
        specification of an argument declaration. Otherwise, returns the same value as `rank`.
        :note: This routine only makes sense in the context of procedure arguments.
        :see: matches_rank
        """
        for expr in bounds:
            if "*" in expr:
                return 1
            elif expr == "..":
                return 0
        return self.rank

    def matches_rank(self,rank):
        """:return: If the given rank matches the rank
        of this record. 
        """
        return rank >= self.min_rank

    def matches_bounds(self,bounds):
        """:return: If the bounds agree.
        """
        # todo: implementation would require knowledge of
        # parameters and equivalency checks between expressions.
        # might be so complex that it must be better moved
        # into semantics module
        assert False, "not implemented"

    @property
    def attributes(self):
        return self.record["attributes"]

    def get_attribute(self,key):
        """:return: The attribute if found, or None.
        :note: In case of 'intent' and 'dimension'
        also the arguments are returned.
        """
        for attrib in self.record["attributes"]:
            if key in ["intent","dimension"]:
                if attrib.startswith(key):
                    return attrib
            elif attrib == key:
                return attrib
        return None 
    def has_attribute(self,key):
        return self.get_attribute(key) != None
    @property
    def is_optional(self):
        return self.get_attribute("optional")
    @property
    def is_parameter(self):
        return self.get_attribute("parameter")
    @property
    def intent(self):
        attrib = self.get_attribute("intent") # whitespaces are removed
        if attrib == "intent(in)":
            return IndexVariable.Intent.IN # can be literal
        elif attrib == "intent(out)":
            return IndexVariable.Intent.OUT # must be variable
        elif attrib == "intent(inout)":
            return IndexVariable.Intent.INOUT # must be variable
        else: # None
            return IndexVariable.Intent.DEFAULT # can be literal

    @property
    def full_type_as_str(self):
        """:return: Fortran string expression for the full data type 
        including kind specification.""" 
        args = []
        if self.kind not in [None,""]:
            args.append("kind="+self.kind)
        if self.len not in [None,""]:
            args.append("len="+self.len)
        if len(args):
            return "{}({})".format(
              self.type, ",".join(args)
            )
        else:
            return self.type

    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element
    @property
    def resolved_kind(self):
        assert self._resolved_kind != None
        return self._resolved_kind
    @property
    def resolved_rhs(self):
        assert self._resolved_rhs != None
        return self._resolved_rhs
    @property
    def resolved_bounds(self):
        assert self._resolved_bounds != None
        return self._resolved_bounds
    @property
    def resolved_size(self):
        assert self._resolved_num_bytes != None
        return self._resolved_num_bytes
    @property
    def resolved_num_bytes(self):
        assert self._resolved_num_bytes != None
        return self._resolved_num_bytes
    
    @resolved_kind.setter
    def resolved_kind(self,resolved_kind):
        self._resolved_kind = resolved_kind
    @resolved_rhs.setter
    def resolved_rhs(self,resolved_rhs):
        self._resolved_rhs = resolved_rhs
    @resolved_bounds.setter
    def resolved_bounds(self,resolved_bounds):
        self._resolved_bounds = resolved_bounds
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element
    @resolved_size.setter
    def resolved_size(self,resolved_size):
        self._resolved_size = resolved_size
    @resolved_num_bytes.setter
    def resolved_num_bytes(self,resolved_num_bytes):
        self._resolved_num_bytes = resolved_num_bytes
    
    @property
    def has_bytes_per_element(self):
        return self._bytes_per_element != None
    @property
    def has_resolved_kind(self):
        return self._resolved_kind != None
    @property
    def has_resolved_rhs(self):
        return self._resolved_rhs != None
    @property
    def has_resolved_bounds(self):
        return self._resolved_bounds != None
    @property
    def has_resolved_size(self):
        return self._resolved_size != None
    @property
    def has_resolved_num_bytes(self):
        return self._resolved_num_bytes != None

    @property
    def module(self):
        return self.record["module"]
    @property
    def is_module_variable(self):
        return self.record["module"] != None
