# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import copy
import tempfile
import shutil

from gpufort import util
from gpufort import translator
from gpufort import indexer

from .. import render

from . import opts

class NamespaceGenerator():
    """This code generator renders a C++
    namespace that contains all or user-selected 
    parameters from a given scope plus additional
    user-prescribed C++ code lines.
    """

    def __init__(self,index,scope_tag,**kwargs):
        r"""Constructor.
        
        :param dict scope: Scope data structure, see "indexer" component.
        :param \*\*kwargs: Keyword arguments.

        :Keyword Arguments:
    
        * *cpp_lines_to_prepend* (`list`):
            Prepend lines to the namespace body.
        * *parameter_filter* (`callable`):
            A filter for the parameters to include, default
            allows all parameters. Takes the arguments (in this order):
            scope tag, Fortran type, Fortran kind, name (lower case), rank.
        * *comment_body* (`bool`):
            Comment out all entries in the body of the namespace.
            This does not affect the lines to prepend or append.
        * *resolve_all_parameters_via_compiler* (`bool`):
            Resolve all parameters via the Fortran compiler.
        * *all_used_modules_have_been_compiled* (`bool`)
            When resolving parameters via Fortran compiler,
            assume that all used modules by the current program/procedure/module
            and its parent(s) (in the procedure case) have been
            compiled already, i.e. there exists a mod file.
            This can only work if every Fortran file that contains a module
            only contains that single module and no other module, program or top-level procedure.
        * *fortran_compiler* (`str`):
            Fortran compiler to use if resolve_all_parameters_via_compiler is True.
        * *fortran_compiler_flags* (`str`):
            Options to pass to the Fortran compiler if resolve_all_parameters_via_compiler is True.
        """
        def default_filter(*args,**kwargs):
            return True
        self.scope = indexer.scope.create_scope(index, scope_tag)
        self.index = index
        self.cpp_lines_to_prepend,_ = util.kwargs.get_value("cpp_lines_to_prepend", opts.prepend_callback(scope_tag), **kwargs)
        self.parameter_filter,_ = util.kwargs.get_value("parameter_filter",opts.parameter_filter, **kwargs)
        self.comment_body,_ = util.kwargs.get_value("comment_body",opts.comment_body, **kwargs)
        self.resolve_all_parameters_via_compiler,_ = util.kwargs.get_value("resolve_all_parameters_via_compiler",
                opts.resolve_all_parameters_via_compiler,**kwargs)
        self.all_used_modules_have_been_compiled,_ = util.kwargs.get_value("all_used_modules_have_been_compiled",opts.all_used_modules_have_been_compiled, **kwargs)
        self.fortran_compiler,_       = util.kwargs.get_value("fortran_compiler",opts.fortran_compiler, **kwargs)
        self.fortran_compiler_flags,_ = util.kwargs.get_value("fortran_compiler_flags",opts.fortran_compiler_flags, **kwargs)

    def __consider_parameter(self,ivar,already_considered):
        return (ivar["rank"] == 0 # todo: remove constraint later on
               and "parameter" in ivar["attributes"]
               and ivar["f_type"] in ["logical","integer","real"]
               and ivar["name"] not in already_considered # hiding, input must be reversed
               and self.parameter_filter(
                    self.scope["tag"],ivar["f_type"],ivar["kind"],ivar["name"],ivar["rank"]))

    def __convert_rhs_to_cpp(self):
        """
        Convert the RHS to of all parameters to a C++ expression.
        """
        body = []
        body += self.cpp_lines_to_prepend 
        already_considered = set()
        for ivar1 in reversed(self.scope["variables"]):
            if self.__consider_parameter(ivar1,already_considered):
                already_considered.add(ivar1["name"])
                ivar = copy.deepcopy(ivar1)
                translator.analysis.append_c_type(ivar)
                # todo: apply post-processing to rhs expression
                #  get rid of selected_real_kind statements and so on
                # todo: consider parameter arrays and complex kinds
                # todo: resolve kind
                tokens = [" "*2]
                if self.comment_body:
                    tokens.append("// ")
                tokens += ["constexpr ",ivar["c_type"]]
                rhs_expr = translator.tree.grammar.arith_logic_expr.parseString(
                            ivar["rhs"],parseAll=True)[0].cstr()
                tokens += [" ",ivar["name"]," = ",rhs_expr,";"]
                body.append("".join(tokens))
        return body

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def __resolve_dependencies(self,all_used_modules_have_been_compiled=False):
        # todo: rewrite docu
        """
        Algorithm:
        given: tag1:tag2:..., = program/procedure tag
        going from inside to outside
        * reproduce hierarchy
        * lookup index record for used modules and filter out all parameters, public and private: create module routine
        OTHER:
        * also copy derived type definitions as there are parameter types
        * ex: type(mytype),parameter :: t = mytype(1) 
        * when rendering, render in order of line orders  
        """
        def create_fortran_construct_context_(index_record):
            """Create an index reocrd like structure based on the original
            index record that only contains parameters and type definitions 
            (latter part not implemented yet).
            """
            nonlocal all_used_modules_have_been_compiled

            context = indexer.create_fortran_construct_record(index_record["kind"], index_record["name"], None)
            context["types"]        += [] # index_record["types"]
            context["variables"]    += [var for var in index_record["variables"] 
                                         if ("parameter" in var["attributes"]
                                            and var["f_type"] != "type")] # todo: consider types too
            type_and_parameter_names = ([var["name"] for var in context["variables"]]
                                       +[typ["kind"] for typ in context["types"]])
            context["accessibility"] = index_record.get("accessibility","public")
            context["public"]        = [ident for ident in index_record.get("public",[]) if ident in type_and_parameter_names]
            context["private"]       = [ident for ident in index_record.get("private",[]) if ident in type_and_parameter_names]
            context["used_modules"] = []
            for used_module1 in index_record["used_modules"]:
                # filter 'only' and rename list to only contain parameters
                if (not all_used_modules_have_been_compiled 
                   and (len(used_module1["renamings"]) 
                       or len(used_module1["only"]))):
                    used_module = copy.deepcopy(used_module1)
                    used_module["renamings"].clear()
                    used_module["only"].clear()
                    for entry in ["only","renamings"]:
                        for pair in used_module1[entry]:
                            try:
                                ivar = indexer.scope.search_index_for_var(
                                        self.index,used_module["name"],
                                        pair["original"])
                                if ("parameter" in ivar["attributes"]
                                   and ivar["f_type"] != "type"):
                                    used_module[entry].append(pair)
                            except util.error.LookupError:
                                pass
                            # todo: also include types
                    context["used_modules"].append(used_module)
                else:
                    context["used_modules"].append(copy.deepcopy(used_module1))
            # extension
            context["declarations"] = []
            iv=0
            it=0
            condv = iv < len(context["variables"])
            condt = it < len(context["types"])
            condition = condv or condt
            while condition:
                if condv and condt:
                    if context["variables"][iv]["lineno"] < context["types"][it]["lineno"]:
                        ivar = context["variables"][iv]
                        context["declarations"].append(indexer.types.render_declaration(ivar))
                        iv += 1
                    else:
                        # todo: render types too
                        #itype = copy.deepcopy(context["variables"][it])
                        #if "device" in itype["attributes"]:
                        #end  
                        it += 1
                elif condv:
                    ivar = context["variables"][iv]
                    context["declarations"].append(indexer.types.render_declaration(ivar))
                    iv += 1
                elif condt:
                    # todo: render types too
                    it += 1
                condv = iv < len(context["variables"])
                condt = it < len(context["types"])
                condition = condv or condt
            return context
        
        modules = []
        module_names = []
        def handle_use_statements_(context):
            """:note:recursive"""
            nonlocal modules
            nonlocal module_names
            nonlocal all_used_modules_have_been_compiled

            for used_module in indexer.scope.combine_use_statements(context["used_modules"]):
                is_third_party_module = ("intrinsic" in used_module["attributes"]
                                         or used_module["name"] in indexer.scope.opts.module_ignore_list)
                if not is_third_party_module:
                    imodule_used = indexer.search_index_for_top_level_entry(
                                    self.index,used_module["name"],["module"])
                    new = create_fortran_construct_context_(imodule_used)
                    handle_use_statements_(new) # recursion
                    if new["name"] not in module_names:
                        modules.append(new)
                        module_names.append(new["name"])


        # Go from tag1 to tag1:tag2:tagn and lookup all index records
        scope_tag_tokens = self.scope["tag"].split(":")
        nesting_levels = len(scope_tag_tokens)
        hierarchy = []
        irecord = None
        for i in range(0,nesting_levels):
            if i == 0:
                irecord = next((ientry for ientry in self.index if ientry["name"] == scope_tag_tokens[0]),None)
            else:
                parent = irecord
                irecord = next((ientry for ientry in parent["procedures"] if ientry["name"] == scope_tag_tokens[i]),None)
            context = create_fortran_construct_context_(irecord)
            if not all_used_modules_have_been_compiled:
                handle_use_statements_(context) # recursive
            hierarchy.append(context)

        return modules, hierarchy

    def __select_parameters_in_scope(self):
        """
        * Exclude parameters that have the same name and have 
          different parent tags. This means they are ambiguous
          and will not be used in a valid code.
        """
        svars_to_resolve = []
        already_considered = set()
        for svar in self.scope["variables"]:
            if svar["name"] not in already_considered:
                if self.__consider_parameter(svar,already_considered):
                    svars_to_resolve.append(svar)
                already_considered.add(svar["name"])
            else: # svar["name"] in already_considered:
                # remove ambiguous entries 
                for existing in copy.copy(svars_to_resolve): # shallow copy
                    if (svar["name"] == existing["name"]
                       and svar["parent_tag"] != existing["parent_tag"]):
                        msg = "ambiguous definition of variable '{}'".format(svar["name"])
                        util.logging.log_warning(opts.log_prefix,"NamespaceGenerator.__select_parameters_in_scope",msg)
                        svars_to_resolve.remove(existing)
        return svars_to_resolve

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def __parse_fortran_output(self,std_out):
        cpp_parameter_expressions = []
        for line in std_out.splitlines():
            #print(line)
            result1,_ =\
                util.parsing.get_top_level_operands(util.parsing.tokenize(line))
            # remove non-printable characters
            result = []
            for column in result1:
                cleaned = "".join([c for c in column if c.isprintable()])
                if cleaned == "<none>" or not len(cleaned.strip()):
                    cleaned == None
                result.append(cleaned)
            #print(result)
            name, f_type, f_len, kind, bpe, rank, sizes, lbounds, rhs_expr = result
            ivar = indexer.types.create_index_var(f_type,f_len,kind,[],name,[],[],rhs_expr)
            translator.analysis.append_c_type(ivar)
            tokens = [" "*2,"constexpr ",ivar["c_type"]]
            tokens += [" ",ivar["name"]," = ",rhs_expr]
            if ivar["c_type"] == "float":
                tokens.append("f")
            tokens.append(";")
            cpp_parameter_expressions.append("".join(tokens))
        return cpp_parameter_expressions

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def __resolve_all_parameters_via_compiler(self):
        ivars_to_resolve = self.__select_parameters_in_scope()
        if len(ivars_to_resolve):
            modules, hierarchy = self.__resolve_dependencies(self.all_used_modules_have_been_compiled)
            #print(ivars_to_resolve)
            fortran_snippet = render.render_resolve_scope_program_f03(
              hierarchy,modules,ivars_to_resolve)
            msg = "scope '{}': snippet for evaluating parameters in scope:\n```\n{}\n```'".format(
                self.scope["tag"],
                fortran_snippet)
            util.logging.log_debug2(opts.log_prefix,"NamespaceGenerator.__resolve_all_parameters_via_compiler",msg)
            #print(fortran_snippet)
            temp_infile = tempfile.NamedTemporaryFile(
                delete=False, 
                mode="w",
                prefix="gpufort-namespacegen",
                suffix=".f90")
            temp_infile.write(fortran_snippet)
            temp_infile.close()
            temp_outfile_path = temp_infile.name.replace(".f90",".x")
            temp_module_dir   = tempfile.TemporaryDirectory(
                prefix="gpufort-namespacegen") # todo: set to True
            # todo: -J,-o assumes gfortran
            cmd_compile = [self.fortran_compiler,"-J" + temp_module_dir.name] + [self.fortran_compiler_flags] + [temp_infile.name,"-o",temp_outfile_path]
            #print(cmd_compile)
            status,_,err_out = util.subprocess.run_subprocess(" ".join(cmd_compile),True)
            if status != 0:
                raise util.error.LookupError("failed resolving parameters in scope '{}' as compilation with compiler '{}' and flags '{}' failed for the following reason: {}".format(
                    self.scope["tag"],self.fortran_compiler," ".join(self.fortran_compiler_flags),err_out))
            # todo: should be cleaned up also in case of error
            shutil.rmtree(temp_module_dir.name,ignore_errors=False)
            if os.path.exists(temp_infile.name):
                os.remove(temp_infile.name)
            cmd_run     = temp_outfile_path
            _,std_out,_ = util.subprocess.run_subprocess(cmd_run,True)
            cpp_parameter_expressions = self.__parse_fortran_output(std_out)
            if os.path.exists(temp_outfile_path):
                os.remove(temp_outfile_path)
            # now fill the namespace body with the resolved parameters
            return cpp_parameter_expressions 
        else:
            return []

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def render_namespace_cpp(self):
        scope_tag = self.scope["tag"]
        result = [ "namespace _{} {{".format(scope_tag.replace(":","_"))]
        if self.resolve_all_parameters_via_compiler:
            result += self.__resolve_all_parameters_via_compiler()  
        else:
            result += self.__convert_rhs_to_cpp()
        result.append("}")
        return result