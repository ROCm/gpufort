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
        return (ivar["rank"] == 0 # TODO remove constraint later on
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
                # TODO apply post-processing to rhs expression
                #  get rid of selected_real_kind statements and so on
                # TODO consider parameter arrays and complex kinds
                # TODO resolve kind
                tokens = [" "*2]
                if self.comment_body:
                    tokens.append("// ")
                tokens += ["constexpr ",ivar["c_type"]]
                rhs_expr = translator.tree.grammar.arithmetic_expression.parseString(
                            ivar["rhs"],parseAll=True)[0].c_str()
                tokens += [" ",ivar["name"]," = ",rhs_expr,";"]
                body.append("".join(tokens))
        return body

    @util.logging.log_entry_and_exit(opts.log_prefix)
    def __resolve_dependencies(self,all_used_modules_have_been_compiled=False):
        """
        Algorithm:

        given: tag1:tag2:..., = program/procedure tag
        going from inside to outside
        1.: determine record mapping to scope tag
           * add use statement for parent tag1_tag2_..tagn before existing use statements: create program routine
           * create program tag1_tag2_.._tagn
           * lookup index record for used modules and filter out all parameters, public and private: create module routine
        2. till n.: determine record mapping to scope tag tag1_tag2...tagn-1
           * add use statement for parent tag_1_tag_...tagn-2 before existing use statements: create module routine
           * create module (!) tag1_tag2...tagn-1
           * lookup index record for used modules and filter out all parameters, public and private: create module routine
        OTHER:
        * also copy derived type definitions as there are parameter types
        * ex: type(mytype),parameter :: t = mytype(1) 
        * when rendering, render in order of line orders  
        ```
        module used1
          use tpl1  
          <public_vars>
          <private_vars>
        end module
     
        module used2
          use tpl2
          <public_vars>
          <private_vars>
        end module

        program tag1
          use used1
          <vars>
          contains
            subroutine tag2
            use used2
            <vars>
            end subroutine
        end program
        ```
        will be translated to
        ```
        module used1
          use tpl1
          <public_vars with 'parameter'>
          <private_vars with 'parameter'>
        end module
     
        module used2
          use tpl2
          <public_vars with 'parameter'>
          <private_vars with 'parameter'>
        end module

        module tag1
          use used1
          public
          <vars with 'parameter'> ! all assumed public
        end module
      
        program tag2
          use tag1
          use used2
          <vars with 'parameter'>
          {for p in <vars in scope tag1:tag2 with 'parameter'>}
          { print properties of <p> }
          { endfor }
        end program
        ```
        """
        # just need the parameters per module program procedure
        # inner most scope tag entity will be translated to program
        # rest will all converted to modules 
        # model program/subroutine parents as modules with all parameters public
        # example: above, subroutine tag2 uses module tag1

        modules = []
        artificial_modules=[]
        program = None
            
        module_names = []

        def create_fortran_construct_context_(kind,index_record,scope_tag_tokens,ignored=[]):
            nonlocal all_used_modules_have_been_compiled

            name = "_".join(scope_tag_tokens) 
            context = indexer.create_fortran_construct_record(kind, name, None)
            context["types"]        += [] # index_record["types"]
            context["variables"]    += [var for var in index_record["variables"] 
                                         if ("parameter" in var["attributes"]
                                            and var["name"] not in ignored
                                            and var["f_type"] != "type")] # TODO consider types too
            type_and_parameter_names = ([var["name"] for var in context["variables"]]
                                       +[typ["kind"] for typ in context["types"]])
            if kind == "module":
                context["accessibility"] = index_record.get("accessibility","public")
                context["public"]        = [ident for ident in index_record.get("public",[]) if ident in type_and_parameter_names]
                context["private"]       = [ident for ident in index_record.get("private",[]) if ident in type_and_parameter_names]
            context["used_modules"] = []
            simple_use_statements_module_names = set()
            for used_module1 in reversed(index_record["used_modules"]):
                used_module = copy.deepcopy(used_module1)
                # only include parameters if we use reproduced models that contain only parameters
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
                            # TODO also include types
                    context["used_modules"].append(used_module)
                else:
                    if used_module["name"] not in simple_use_statements_module_names:
                        simple_use_statements_module_names.add(used_module["name"])
                        context["used_modules"].append(used_module)
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
                        # TODO render types too
                        #itype = copy.deepcopy(context["variables"][it])
                        #if "device" in itype["attributes"]:
                        #end  
                        it += 1
                elif condv:
                    ivar = context["variables"][iv]
                    context["declarations"].append(indexer.types.render_declaration(ivar))
                    iv += 1
                elif condt:
                    # TODO render types too
                    it += 1
                condv = iv < len(context["variables"])
                condt = it < len(context["types"])
                condition = condv or condt
            return context


        # 1) Go from tag1 to tag1:tag2:tagn and lookup all index records
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
            hierarchy.append(irecord)        
        # 2) Go from tag1:tag2:tagn to tag1
        # To have original top down order, all artificial modules must be prepended
        # to the corresponding list.
        names_used_by_inner_contexts = []
        for i,irecord in enumerate(reversed(hierarchy)):
            make_program = i == 0 
            kind = "program" if make_program else "module"
            context = create_fortran_construct_context_(kind,irecord,scope_tag_tokens[:nesting_levels-i],
                                                         ignored=names_used_by_inner_contexts)
            # While a procedure belonging to a program/module/procedure can hide the parent's definitions,
            # a program/module/procedure cannot hide definitions included from a module.
            # In this case, Fortran compilers emit an error stating that a name is used twice.
            # As we convert outer program/module/procedures to artificial modules,
            # we need to exclude names that have already be used by inner procedures.
            names_used_by_inner_contexts += ([var["name"] for var in context["variables"]]
                                              +[var["name"] for var in context["types"]])
            if make_program:
                program = context
            else:
                if context["name"] not in module_names:
                    artificial_modules.insert(0,context) # prepend to invert order of list
                    module_names.append(context["name"])
        # 3) Now go again from tag1 to tag1:tag2:tagn and handle the use statements
        # all used modules must be appended to the corresponding list
        contexts = artificial_modules+[program] 
        def handle_use_statements_(context):
            """:note:recursive"""
            nonlocal modules
            nonlocal all_used_modules_have_been_compiled

            for used_module in indexer.scope.combine_use_statements(context["used_modules"]):
                is_third_party_module = ("intrinsic" in used_module["attributes"]
                                         or used_module["name"] in indexer.scope.opts.module_ignore_list)
                if not is_third_party_module:
                    imodule_used = next((ientry for ientry in self.index if ientry["name"] == used_module["name"]),None)
                    if imodule_used != None:
                        new = create_fortran_construct_context_("module",imodule_used,[used_module["name"]])
                        handle_use_statements_(new) # recursion
                        if new["name"] not in module_names:
                            modules.append(new)
                            module_names.append(new["name"])
                    else:
                        msg = "no index record found for module '{}'".format(
                            used_module["name"])
                        raise util.error.LookupError(msg)
        # 4) Handle existing use statements
        if not all_used_modules_have_been_compiled:
            for context in contexts:
                handle_use_statements_(context) # recursive
        # 5) Then add additional predefined ones (parent, iso_c_binding, ...)
        # Loop cannot be combined with above loop as above loop would try to
        # lookup index record of artificially generated modules, which
        # are not part of the index
        for i,context in enumerate(contexts):
            if i > 0:
                parent_name = contexts[i-1]["name"]
                context["used_modules"].insert(0,indexer.create_index_record_from_use_statement("use {}".format(parent_name)))
        program["used_modules"].insert(0,indexer.create_index_record_from_use_statement("use iso_c_binding"))
        program["used_modules"].insert(1,indexer.create_index_record_from_use_statement("use iso_fortran_env"))

        return modules+artificial_modules, program

    def __select_parameters_in_scope(self):
        ivars_to_resolve = []
        already_considered = set()
        for ivar1 in reversed(self.scope["variables"]):
            ivar = copy.deepcopy(ivar1)
            # public,private may only appear in module
            try:
                ivar["attributes"].remove("public")
            except:
                pass 
            try:
                ivar["attributes"].remove("private")
            except:
                pass 
            if self.__consider_parameter(ivar,already_considered):#
                already_considered.add(ivar["name"])
                ivars_to_resolve.insert(0,ivar)
        return ivars_to_resolve

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
        decl_list = []
        ivars_to_resolve = self.__select_parameters_in_scope()
        modules, program = self.__resolve_dependencies()
        #print(decl_list)
        #print(ivars_to_resolve)
        fortran_snippet = render.render_resolve_scope_program_f03(
          modules,program,ivars_to_resolve)
        msg = "scope '{}': snippet for evaluating parameters in scope:\n```\n{}\n```'".format(
            self.scope["tag"],
            fortran_snippet)
        util.logging.log_debug2(opts.log_prefix,"NamespaceGenerator.__resolve_all_parameters_via_compiler",msg)
        temp_infile = tempfile.NamedTemporaryFile(
            delete=False, 
            mode="w",
            prefix="gpufort-namespacegen",
            suffix=".f90")
        temp_infile.write(fortran_snippet)
        temp_infile.close()
        temp_outfile_path = temp_infile.name.replace(".f90",".x")
        temp_module_dir   = tempfile.TemporaryDirectory(
            prefix="gpufort-namespacegen") # TODO set to True
        # TODO -J,-o assumes gfortran
        cmd_compile = [self.fortran_compiler,"".join(["-J",temp_module_dir.name])] + self.fortran_compiler_flags + [temp_infile.name,"-o",temp_outfile_path]
        #print(cmd_compile)
        status,_,err_out = util.subprocess.run_subprocess(cmd_compile,True)
        if status != 0:
            raise util.error.LookupError("failed resolving parameters in scope '{}' as compilation with compiler '{}' and flags '{}' failed for the following reason: {}".format(
                self.scope["tag"],self.fortran_compiler," ".join(self.fortran_compiler_flags),err_out))
        # TODO should be cleaned up also in case of error
        shutil.rmtree(temp_module_dir.name,ignore_errors=False)
        if os.path.exists(temp_infile.name):
            os.remove(temp_infile.name)
        cmd_run     = [temp_outfile_path]
        _,std_out,_ = util.subprocess.run_subprocess(cmd_run,True)
        cpp_parameter_expressions = self.__parse_fortran_output(std_out)
        if os.path.exists(temp_outfile_path):
            os.remove(temp_outfile_path)
        # now fill the namespace body with the resolved parameters
        return cpp_parameter_expressions 

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
