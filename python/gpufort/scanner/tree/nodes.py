# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re
import sys
import hashlib
import textwrap

from gpufort import util
from gpufort import linemapper
from gpufort import translator
from gpufort import indexer

from .. import opts
from . import grammar
from . import backends

SCANNER_ERROR_CODE = 1000

p_attributes = re.compile(r"attributes\s*\(\s*\w+\s*(,\s*\w+)?\s*\)\s*",
                          flags=re.IGNORECASE)


def remove_type_prefix(var_name):
    return var_name.split("%")[-1]


def replace_ignore_case(key, subst, text):
    return re.sub(re.escape(key), subst, text, flags=re.IGNORECASE)


def flatten_list(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_list(x):
                yield sub_x
        else:
            yield x


# scanner tree
class STNode:

    def __init__(self, first_linemap, first_statement_index=0):
        self.name = None
        self.kind = None
        self._linemaps = []
        if first_linemap != None:
            self._linemaps.append(first_linemap)
        self._first_statement_index = first_statement_index
        self._last_statement_index = first_statement_index # inclusive
        # metadata
        self.parent = None
        self.children = []
        self.ignore_in_s2s_translation = False

    def __get_linemaps_content(self,
                               key,
                               first_linemap_first_elem=0,
                               last_linemap_last_elem=-1):
        """Collects entries for the given key from the node's linemaps."""
        return linemapper.get_linemaps_content(self._linemaps, key,
                                               first_linemap_first_elem,
                                               last_linemap_last_elem)

    def add_linemap(self, linemap):
        """Adds a linemap if it differs from the last linemap."""
        if not len(self._linemaps
                  ) or self._linemaps[-1]["lineno"] < linemap["lineno"]:
            self._linemaps.append(linemap)

    def complete_init(self, index=[]):
        """Complete the initialization
        
        This routine is called after all associated linemaps have
        been added to this node.
        """
        pass

    def remove_comments(self, lines):
        """Remove comments but keep directives."""
        # TODO move somewhere else
        for line in list(lines): # shallow copy
            stripped_line = line.lstrip("\t ")
            if not len(stripped_line) or\
               (stripped_line[0] in ["*","c","C","!"] and not\
               stripped_line[1] == "$"):
                lines.remove(line)

    def remove_whitespaces(self, lines):
        """Remove any whitespace and line continuation characters."""
        for i, line in enumerate(lines):
            lines[i] = line.replace(" ", "").replace("\t", "").replace(
                "\n", "").replace("&", "")

    def lines(self):
        original_lines = self.__get_linemaps_content("lines")
        return util.parsing.relocate_inline_comments(original_lines)

    def statements(self, include_none_entries=False):
        """Extract the statements associated with this node from the linemaps associated with this node.
        :param bool include_none_entries: Also include entries that are None [default=False].
        :note: None entries might have been introduced by other nodes transforming the same linemap(s).
        """
        statements_of_node = self.__get_linemaps_content("statements",\
          self._first_statement_index,self._last_statement_index)
        result = []
        if len(statements_of_node):
            for stmt in statements_of_node:
                if stmt["body"] != None:
                    result.append(stmt["body"].rstrip("\n\t ;"))
                elif include_none_entries:
                    result.append(None)
        return result

    def min_lineno(self):
        """
        :return: Inclusive first line number belonging to this object.
        """
        return self._linemaps[0]["lineno"]

    def max_lineno(self):
        """
        :return: Inclusive last line number belonging to this object.
        """
        last_linemap = self._linemaps[-1]
        return last_linemap["lineno"] + len(last_linemap["lines"]) - 1

    def first_line(self):
        """
        :return: First line in first linemap.
        """
        return self._linemaps[0]["lines"][0]

    def first_line_indent(self):
        """Indent chars at beginning of first line."""
        first_line = self.first_line()
        num_indent_chars = len(first_line) - len(first_line.lstrip(" \t"))
        return first_line[0:num_indent_chars]

    def first_statement(self):
        """
        :return: First statement in first linemap that belongs to this node.
        """
        return self._linemaps[0]["statements"][
            self._first_statement_index]["body"]

    def append(self, child):
        self.children.append(child)

    def list_of_parents(self):
        """
        Returns a list that contains all
        parents of a node plus the node itself._
        """
        result = []

        def recursive_parent_lookup(curr):
            if curr != None:
                result.append(curr)
                recursive_parent_lookup(curr.parent)

        recursive_parent_lookup(self)
        return result

    def find_all(self, filter=lambda child: True, recursively=False):
        result = []

        def descend_(curr):
            nonlocal result
            for child in curr.children:
                if filter(child):
                    result.append(child)
                if recursively:
                    descend_(child)

        descend_(self)
        return result

    def find_first(self, filter=lambda child: True):
        """:return: First child that matches the filter, or None.
        :note: Not recursively applied to children.
        """
        for child in self.children:
            if filter(child):
                return child
        return None

    def find_last(self, filter=lambda child: True):
        """:return: Last child that matches the filter, or None.
        :note: Not recursively applied to children.
        """
        for child in reversed(self.children):
            if filter(child):
                return child
        return None

    def extract_first(self, text, expression):
        """
        Extract (first) part of the text that contains a given expression.
        :rtype: str
        :return: The first part of the text that matches the expression.
        """
        for tokens, start, end in expression.scanString(text):
            return text[start:end]
        return ""

    def extract_all(self, text, expression):
        """Extract all parts of the text that contain a given pyparsing expression.
        :rtype: list
        :return: All parts of the text that match the pyparsing expression
        """
        result = []
        for tokens, start, end in expression.scanString(text):
            result.append(text[start:end])
        return result

    def get_prolog(self):
        """:return: the first statement's prolog or the
        first linemap's prolog if the first statement is
        the first statement in the first linemap.
        """
        prolog = self._linemaps[0]["statements"][
            self._first_statement_index]["prolog"]
        if self._first_statement_index == 0:
            prolog = self._linemaps[0]["prolog"]
        return prolog
          
    def get_epilog(self):
        """:return: the last statement's epilog or the
        last linemap's epilog if the last statement is
        the last statement in the last linemap.
        """
        epilog = self._linemaps[-1]["statements"][
            self._last_statement_index]["epilog"]
        have_last_in_last_linemap = self._last_statement_index  == -1 or\
                                    self._last_statement_index  == len(self._linemaps[-1]["statements"])-1
        if have_last_in_last_linemap:
            epilog = self._linemaps[0]["epilog"]
        return epilog

    def add_to_prolog(self, line, prepend=False):
        """Add some prolog lines to the first linemap."""
        line_preproc = line.rstrip()
        prolog = self.get_prolog() 
        if self._first_statement_index > 0:
            self._linemaps[0]["modified"] = True
        if not line_preproc.lower() in map(str.lower, prolog):
            if prepend:
                prolog.insert(0, line_preproc)
            else:
                prolog.append(line_preproc)

    def add_to_epilog(self, line, prepend=False):
        """Add some epilog lines to the first linemap."""
        line_preproc = line.rstrip()
        epilog = self.get_epilog() 
        have_last_in_last_linemap = self._last_statement_index  == -1 or\
                                    self._last_statement_index  == len(self._linemaps[-1]["statements"])-1
        if not have_last_in_last_linemap:
            self._linemaps[0]["modified"] = True
        if not line_preproc.lower() in map(str.lower, epilog):
            if prepend:
                epilog.insert(0, line_preproc)
            else:
                epilog.append(line_preproc)

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        """Transforms statements associated with underlying linemaps (hook)
        :param line: An excerpt from a Fortran file, possibly multiple lines
        :type line: str
        :return: If the text was changed at all
        :rtype: bool
        """
        if statements_fully_cover_lines:
            return joined_lines, False
        else:
            return joined_statements, False

    def _modify_linemaps(self, substitution):
        """
        Replaces first statement associated with node in first associated linemap by 'subst' argument.
        Replaces all other other statements in first and all other linemaps associated
        with node by 'None'. Marks all linemaps associated with this node as modified.
        :param str subst: The text that should be written in first associated statement in first
                          associated linemap.
        :note: Post processing working with the modified statements must ignore all 'None' statements.
        :note: We assume that the statements of an expression that spreads over multiples lines are only modified once.
        :note: We write 'None' entries into the statements instead of clipping them away because 
               if multiple statements per line are present and other nodes modify those, removing elements from the list
               of statements might mess with the indexing.
        """
        first_linemap_first_elem = self._first_statement_index
        last_linemap_last_elem = self._last_statement_index
        # write subst into first linemap first statement
        self._linemaps[0]["modified"] = True
        self._linemaps[0]["statements"][first_linemap_first_elem][
            "body"] = substitution
        assert len(self._linemaps), "self._linemaps should not be empty"
        last_linemap_ubound = last_linemap_last_elem
        if last_linemap_ubound != -1:
            last_linemap_ubound += 1

        def assign_none_(statements, lbound=0, ubound=-1):
            if ubound == -1:
                ubound = len(statements)
            for i in range(lbound, ubound):
                statements[i]["body"] = None

        if len(self._linemaps) == 1:
            assign_none_(self._linemaps[0]["statements"],
                         first_linemap_first_elem + 1, last_linemap_ubound)
        else:
            self._linemaps[-1]["modified"] = True
            assign_none_(self._linemaps[0]["statements"],
                         first_linemap_first_elem + 1)
            assign_none_(self._linemaps[-1]["statements"], 0,
                         last_linemap_ubound)
            for linemap in self._linemaps[1:-1]: # upper bound exclusive
                linemap["modified"] = True
                assign_none_(linemap["statements"])

    def transform_statements(self, index=[]):
        """
        Replaces original statements by generated code. Modifies the 'statements' 
        epilog and prolog entries of the associated linemaps.
        :param list index: Index dictionary containing indexer modules/programs/top-level procedures
               as entries.
        :note: When multiple linemaps contain the expression associated with this note,
               the transformed code is written into the first associated statement in 
               the first linemap and the remaining associated statements in the first
               and all other linemaps are replaced by None.
        :note: If the node's first statement in a linemap, the node's prolog
               lines are appended to the linemaps prolog field. Analogously,
               the epilog is appended to the linemaps epilog field if
               the node's last statement is the last statement in 
               the last line map. Otherwise, epilog / prolog are appended/prepended
               directly to the statement.
        """
        if not self.ignore_in_s2s_translation:
            have_first_in_first_linemap = self._first_statement_index == 0
            have_last_in_last_linemap     = self._last_statement_index  == -1 or\
                                            self._last_statement_index  == len(self._linemaps[-1]["statements"])-1
            joined_statements = "\n".join(self.statements())
            joined_lines = "".join(self.lines())
            statements_fully_cover_lines = have_first_in_first_linemap and have_last_in_last_linemap
            transformed_code, transformed = \
              self.transform(joined_lines,joined_statements,statements_fully_cover_lines,index)
            if transformed:
                self._modify_linemaps(transformed_code)


class IDeclListEntry:
    pass


class STContainerBase(STNode):

    @staticmethod
    def decl_list_entry_filter(stnode):
        """:return: If the scanner tree node is member of the declaration list."""
        return isinstance(stnode, IDeclListEntry)

    def add_use_statement(self,module,only=[]):
        """Add a use statement to the top of
        the module's declaration list.
        :param str module: Module name
        :param list only: 
        """
        indent_parent = self.first_line_indent()
        indent = indent_parent + " "*2
        statement = "{}use {}".format(indent,module)
        if len(only):
            statement += ", only: "+",".join(only)
        self.add_to_epilog(statement,prepend=True)

    def first_entry_in_decl_list(self):
        result = self.find_first(filter=STContainerBase.decl_list_entry_filter)
        if result == None:
            return self
        return result

    def last_entry_in_decl_list(self):
        result = self.find_last(filter=STContainerBase.decl_list_entry_filter)
        if result == None:
            return self
        return result

    def return_or_end_statements(self):
        return self.find_all(
            filter=lambda stnode: isinstance(stnode, STEndOrReturn))

    def end_statement(self):
        """:return: The child of type STEnd."""
        return self.find_last(filter=lambda stnode: isinstance(stnode, STEnd))

    def has_contains_statement(self):
        """:return: A child of type STContains, or None if no such child node can be found."""
        return self.find_first(
            filter=lambda stnode: isinstance(stnode, STContains)) != None

    def append_to_decl_list(self, lines, prepend=False):
        """Append lines to the last statement in the declaration list.
        :param list lines: The lines to append.
        :param bool prepend: Prepend new lines to previously append lines.
        """
        last_decl_list_node = self.last_entry_in_decl_list()
        indent = last_decl_list_node.first_line_indent()
        if last_decl_list_node is None:
            last_decl_list_node = self
        for line in lines:
            last_decl_list_node.add_to_epilog("".join([indent, line]), prepend)

    def append_vars_to_decl_list(self, varnames, vartype="type(c_ptr)"):
        """Create and add declaration expression from `varnames` and `vartype` to declaration list 
           if they are not part of the declartion list yet."""
        if len(varnames):
            self.append_to_decl_list(\
              [" :: ".join([vartype,name]) for name in varnames],prepend=True)

    def prepend_to_return_or_end_statements(self, lines):
        """
        Prepend lines to the last statement in the declaration list.
        :param list lines: The lines to append.
        :param bool prepend: Prepend new lines to previously append lines.
        """
        for return_or_end_node in self.return_or_end_statements():
            indent = return_or_end_node.first_line_indent()
            for line in lines:
                return_or_end_node.add_to_prolog(indent + line)

    def tag(self):
        """Construct a tag that can be used to search the index."""
        result = self.name.lower()

        def recursive_parent_lookup(curr):
            nonlocal result
            if type(curr) != STRoot:
                result = curr.name.lower() + ":" + result
                recursive_parent_lookup(curr.parent)

        recursive_parent_lookup(self.parent)
        return result

    def local_and_dummy_var_names(self, index):
        """:return: names of local variables (1st retval) and dummy variables (2nd retval)."""
        # TODO can also be realised by subclass
        scope = indexer.scope.create_scope(index, self.tag())
        scope_vars = scope["variables"]
        if type(self) == STProgram:
            irecord = next(
                (ientry for ientry in index if ientry["name"] == self.name),
                None)
            dummy_arg_names = []
        else:
            parent_tag = self.parent.tag()
            irecord = indexer.scope.search_index_for_procedure(
                index, parent_tag, self.name)
            dummy_arg_names = irecord["dummy_args"]
        local_var_names = [
            ivar["name"]
            for ivar in irecord["variables"]
            if ivar["name"] not in dummy_arg_names
        ]
        return local_var_names, dummy_arg_names


class STEndOrReturn(STNode):

    def transform_statements(self, index=[]):
        prolog = self._linemaps[0]["prolog"]
        if len(prolog):
            joined_statements = "\n".join(self.statements())
            indent = joined_statements[0:len(joined_statements)
                                       - len(joined_statements.lstrip())]
            joined_prolog = "\n".join([indent + l.lstrip() for l in prolog
                                      ]).rstrip()
            transformed_code = joined_prolog + "\n" + joined_statements
            self._modify_linemaps(transformed_code)
        self._linemaps[0]["prolog"].clear()


class STEnd(STEndOrReturn):
    pass


class STReturn(STEndOrReturn):
    pass


class STRoot(STContainerBase):

    def __init__(self):
        STNode.__init__(self, None, -1)

    def tag(self):
        return None


class STModule(STContainerBase):

    def __init__(self, name, first_linemap, first_linemap_first_statement):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self.name = name.lower()
        self.kind = "module"


class STProgram(STContainerBase):

    def __init__(self, name, first_linemap, first_linemap_first_statement):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self.name = name.lower()
        self.kind = "program"

class STContains(STNode):
    pass

class STPlaceHolder(STNode, IDeclListEntry):
    pass

class STProcedure(STContainerBase):

    def __init__(self, name, parent_tag, kind, first_linemap,
                 first_linemap_first_statement, index):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self.name = name
        self.kind = kind
        self.code = []
        # check attributes
        self.index_record = indexer.scope.search_index_for_procedure(
            index, parent_tag, name)
        self.kernel_args_tavars = []
        self.c_result_type = "void"
        self.parse_result = None

    def __must_be_available_on_host(self):
        return not len(self.index_record["attributes"]) or\
               "host" in self.index_record["attributes"]

    def __attributes_present(self):
        return len(self.index_record["attributes"])

    def complete_init(self, index):
        # exclude begin/end statement of procedure
        self.code = self.statements()[1:-1] # implies copy
        scope = indexer.scope.create_scope(index, self.tag())
        iprocedure = self.index_record

        try:
            if self.is_function():
                result_name = iprocedure["result_name"]
                ivar_result = next([
                    var for var in iprocedure["variables"]
                    if var["name"] == iprocedure["result_name"]
                ], None)
                if ivar_result != None:
                    self.c_result_type = ivar_result["c_type"]
                    # TODO catch errors here
                    # statements must contain filename and line numbers as well
                    self.parse_result = translator.parse_procedure_body(
                        self.code, scope, ivar_result["name"])
                else:
                    raise util.error.LookupError("could not identify return value for function ''")
            else:
                self.c_result_type = "void"
                # TODO catch errors here
                # statements must contain filename and line numbers as well
                self.parse_result = translator.parse_procedure_body(
                    self.code, scope)
        except util.error.SyntaxError as e:
            raise util.error.SyntaxError("{}:{}:{}".format(
                    self._linemaps[0]["file"],self._linemaps["0"]["lineno"],e.msg)) from e
        except util.error.LimitationError as e:
            raise util.error.LimitationError("{}:{}:{}".format(
                    self._linemaps[0]["file"],self._linemaps["0"]["lineno"],e.msg)) from e

    def is_function(self):
        """:return: If the procedure is a function. Otherwise, it is a subroutine."""
        return self.index_record["kind"] == "function"

    def has_attribute(self, attribute):
        return attribute in self.index_record["attributes"]

    def is_kernel_subroutine(self):
        return self.has_attribute("global")

    def must_be_available_on_device(self):
        return self.has_attribute("device") or\
               self.has_attribute("global")

    def keep_recording(self):
        """
        No recording if the function needs to be kept only on the host.
        """
        return self.must_be_available_on_device()

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        """
        Treats CUF and OpenACC subroutines/functions that carry CUF-specific attributes
        or require a device version as specified in a directive.

        :note: Removes 'attributes(...)' objects from the procedure header
        when encountering a CUF device procedure that
        needs to be kept on the host too.
        """
        attributes_present = self.__attributes_present()
        must_be_available_on_device = self.must_be_available_on_device()
        must_be_available_on_host = self.__must_be_available_on_host()

        original = joined_lines
        if attributes_present: # CUF case
            if must_be_available_on_host:
                return p_attributes.sub("", original), True
            elif must_be_available_on_device: # and not must_be_available_on_host
                indent = self.first_line_indent()
                return "{0}! extracted to HIP C++ file".format(indent), True
        else:
            return original, False


class STDirective(STNode):

    def __init__(self,
                 first_linemap,
                 first_linemap_first_statement,
                 directive_no,
                 sentinel="!$cuf"):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self._sentinel = sentinel
        self._directive_no = directive_no


class STLoopNest(STNode):

    def __init__(self, *args, **kwargs):
        STNode.__init__(self, *args, **kwargs)
        self._do_loop_ctr_memorised = -1
        self.__hash = None
        #
        self.parse_result          = None
        self.sharedmem_f_str       = "0" # set from extraction routine
        self.stream_f_str          = "c_null_ptr" # set from extraction routine
        self.async_launch_f_str = ".false."
        #
        self.kernel_args_tavars = [] # set from extraction routine
        self.kernel_args_names = [] # set from subclass
        self.code = []
    def __hash_kernel(self):
        """Compute hash code for this kernel. Must be done before any transformations are performed."""
        statements = list(self.code) # copy
        self.remove_comments(statements)
        self.remove_whitespaces(statements)
        snippet = "".join(statements)
        return hashlib.md5(snippet.encode()).hexdigest()[0:6]

    def complete_init(self, index=[]):
        self.code = self.statements()
        self.__hash = self.__hash_kernel()
        parent_tag = self.parent.tag()
        scope = indexer.scope.create_scope(index, parent_tag)
        try:
            self.parse_result = translator.parse_loop_kernel(self.code, scope)
        except util.error.SyntaxError as e:
            raise util.error.SyntaxError("{}:{}:{}".format(
                    self._linemaps[0]["file"],self._linemaps["0"]["lineno"],e.msg)) from e
        except util.error.LimitationError as e:
            raise util.error.LimitationError("{}:{}:{}".format(
                    self._linemaps[0]["file"],self._linemaps["0"]["lineno"],e.msg)) from e

    def kernel_name(self):
        """derive a name for the kernel"""
        return opts.loop_kernel_name_template.format(
            parent=self.parent.name.lower(),
            lineno=self.min_lineno(),
            hash=self.__hash)

    def kernel_hash(self):
        return self.__hash

    def kernel_launcher_name(self):
        return "launch_{}".format(self.kernel_name())

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if opts.destination_dialect.startswith("hip"):
            self.parent.add_use_statement("gpufort_array")
            self.parent.add_use_statement("hipfort_types",only=["dim3"])
            #
            kernel_args = []
            # determine grid or problem size
            launcher_name = self.kernel_launcher_name()
            launcher_name_suffix = "_hip"
            grid_or_ps_f_str  = self.parse_result.grid_expr_f_str()
            if grid_or_ps_f_str == None and self.parse_result.num_gangs_teams_blocks_specified():
                grid = self.parse_result.num_gangs_teams_blocks()
                grid_or_ps_f_str = "dim3({})".format(",".join(grid))
            elif grid_or_ps_f_str == None:
                launcher_name_suffix = "_hip_ps"
                grid_or_ps_f_str = "dim3({})".format(",".join(self.parse_result.problem_size()))
            launcher_name += (launcher_name_suffix if not opts.loop_kernel_default_launcher=="cpu" else "_cpu_")
            ## determine block size
            block_f_str = self.parse_result.block_expr_f_str()
            if block_f_str == None and self.parse_result.num_threads_in_block_specified():
                block = self.parse_result.num_threads_in_block()
                block_f_str = "dim3({})".format(",".join(block))
            elif block_f_str == None:
                block_f_str = "dim3(128)" # use config values 
            kernel_args.append(grid_or_ps_f_str)
            kernel_args.append(block_f_str)
            kernel_args.append(self.sharedmem_f_str)
            # stream
            try:
                stream_as_int = int(self.stream_f_str)
                stream = "c_null_ptr"
            except:
                stream = self.stream_f_str
            kernel_args.append(stream)
            kernel_args.append(self.async_launch_f_str)
            kernel_args += self.kernel_args_names
            #
            result = []
            result.append("! extracted to HIP C++ file\n")
            result.append("call {0}(&\n  {1})\n".format(launcher_name,",&\n  ".join(kernel_args)))
            indent = self.first_line_indent()
            return textwrap.indent("".join(result),indent), True
        else:
            return None, False

class IWithBackend:
    @classmethod
    def register_backend(cls, src_dialect, dest_dialects, func):
        cls._backends.append((src_dialect, dest_dialects, func))

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        """Applies the registered backends's actions to 
        the joined statements if a backend's source dialect is in the list
        of source dialects that should be translated. Further checks if the
        backend's destination dialect is a substring(!) of the specified destination dialect.

        :param str joined_lines: The original lines joined to a string.
        :param str joined_statements: The (pre-processed) statements derived from the original lines
                                      joined to a string.
        :param bool statements_fully_cover_lines: If the lines do not contain unrelated
                                                  statements at the begin of the first line or end of the last line.
        :return: A tuple of the transformed string and if it differs from the original.
        """
        result = joined_statements
        transformed = False
        for src_dialect, dest_dialects, func in self.__class__._backends:
            if (src_dialect in opts.source_dialects and
                  opts.destination_dialect in dest_dialects):
                result, transformed1 = func(self, result, index)
                transformed = transformed or transformed1
        return result, transformed

class STDeclaration(STNode, IWithBackend, IDeclListEntry):
    """Represents Fortran declarations.
    ```
    """
    _backends = []
    
    def __init__(self,*args,**kwargs):
        STNode.__init__(self,*args,**kwargs)

    def transform(self,*args,**kwargs):
        try:
            return IWithBackend.transform(self,*args,**kwargs)
        except util.error.SyntaxError as e:
            first_linemap = self._linemaps[0]
            filepath = first_linemap["file"]
            lineno=first_linemap["lineno"]
            statement_no  = self._first_statement_index
            msg = "{}:{}:{}(stmt-no):{}".format(filepath,lineno,current_statement_no+1,str(e))
            raise util.error.SyntaxError(msg) from e
def index_var_is_on_device(ivar):
    return "device" in ivar["qualifiers"]


def pinned_or_on_device(ivar):
    """:return: Qualifier and if special treatment is necessary."""
    if "device" in ivar["qualifiers"]:
        return "device", True
    elif "pinned" in ivar["qualifiers"]:
        return "pinned", True
    else:
        return None, False


class STNonZeroCheck(STNode):
    
    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]): # TODO
        result = snippet
        transformed = False
        for tokens, start, end in translator.tree.grammar.non_zero_check.scanString(
                result):
            parse_result = tokens[0]
            lhs_name = parse_result.lhs_f_str()
            ivar  = indexer.scope.search_index_for_var(index,self.parent.tag(),\
              lhs_name)
            on_device = index_var_is_on_device(ivar)
            transformed |= on_device
            if on_device:
                subst = parse_result.f_str() # TODO backend specific
                result = result.replace(result[start:end], subst)
        return result, transformed


class STAllocated(STNode):

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]): # TODO

        def repl(parse_result):
            var_name = parse_result.var_name()
            ivar = indexer.scope.search_index_for_var(index,self.parent.tag(),\
              var_name)
            on_device = index_var_is_on_device(ivar)
            return (parse_result.f_str(), on_device) # TODO backend specific

        result, transformed = util.pyparsing.replace_all(
            joined_statements, translator.tree.grammar.allocated, repl)
        assert result != None
        return result, transformed

class STUseStatement(STNode, IDeclListEntry, IWithBackend):
    _backends = []
  
    def __init__(self,*args,**kwargs):
        STNode.__init__(self,*args,**kwargs)
    
    def transform(self,*args,**kwargs):
        return IWithBackend.transform(self,*args,**kwargs)

class STAllocate(STNode, IWithBackend):
    _backends = []

    def __init__(self, first_linemap, first_linemap_first_statement):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self.parse_result = translator.tree.grammar.allocate.parseString(
            self.statements()[0])[0]
        self.variable_names = self.parse_result.variable_names()
    
    def transform(self,*args,**kwargs):
        return IWithBackend.transform(self,*args,**kwargs)


class STDeallocate(STNode, IWithBackend):
    _backends = []

    def __init__(self, first_linemap, first_linemap_first_statement):
        STNode.__init__(self, first_linemap, first_linemap_first_statement)
        self.parse_result = translator.tree.grammar.deallocate.parseString(
            self.statements()[0])[0]
        self.variable_names = self.parse_result.variable_names()
    
    def transform(self,*args,**kwargs):
        return IWithBackend.transform(self,*args,**kwargs)

class STMemcpy(STNode):

    def __init__(self, *args, **kwargs):
        STNode.__init__(self, *args, **kwargs)
        self._parse_result = translator.tree.grammar.memcpy.parseString(
            self.statements()[0])[0]

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]): # TODO backend specific
        indent = self.first_line_indent()
        def repl_memcpy_(parse_result):
            dest_name = parse_result.dest_name_f_str()
            src_name = parse_result.src_name_f_str()
            try:
                dest_indexed_var = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                  dest_name)
                src_indexed_var  = indexer.scope.search_index_for_var(index,self.parent.tag(),\
                  src_name)
                dest_on_device = index_var_is_on_device(dest_indexed_var)
                src_on_device = index_var_is_on_device(src_indexed_var)
            except util.error.LookupError:
                dest_on_device = False 
                src_on_device  = False
            if dest_on_device or src_on_device:
                subst = parse_result.hip_f_str(dest_on_device, src_on_device)
                return (textwrap.indent(subst,indent), True)
            else:
                return ("", False) # no transformation; will not be considered

        return repl_memcpy_(self._parse_result)
        #return util.pyparsing.replace_all(joined_statements,translator.tree.grammar.memcpy,repl_memcpy)
