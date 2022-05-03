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
    def __resolve_dependencies(self):
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
        module_names = []
        program = None

        def append_module(irecord):
            if irecord["name"] not in module_names:
                modules.append(irecord)
                module_names.append(irecord["name"])

        # given: tag1:tag2:..., = program/procedure tag
        # going from inside to outside
        # 1.: determine record mapping to scope tag
        #    * add use statement for parent tag1_tag2_..tagn before existing use statements: create program routine
        #    * create program tag1_tag2_.._tagn
        #    * lookup index record for used modules and filter out all parameters, public and private: create module routine
        # 2. till n.: determine record mapping to scope tag tag1_tag2...tagn-1
        #    * add use statement for parent tag_1_tag_...tagn-2 before existing use statements: create module routine
        #    * create module (!) tag1_tag2...tagn-1
        #    * lookup index record for used modules and filter out all parameters, public and private: create module routine
        # OTHER:
        # * also copy derived type definitions as there are parameter types
        # * ex: type(mytype),parameter :: t = mytype(1) 
        # * when rendering, render in order of line orders  
        def create_fortran_construct_record_(kind,index_record,scope_tag_tokens):
            name = "_".join(scope_tag_tokens) 
            construct = indexer.create_fortran_construct_record(kind, name, None)
            construct["types"]        += [] # index_record["types"]
            construct["variables"]    += [var for var in index_record["variables"] 
                                         if ("parameter" in var["attributes"]
                                            and var["f_type"] != "type")] # TODO consider types too
            type_and_parameter_names = ([var["name"] for var in construct["variables"]]
                                       +[typ["kind"] for typ in construct["types"]])
            if kind == "module":
                construct["accessibility"] = index_record.get("accessibility","public")
                construct["public"]        = [ident for ident in index_record.get("public",[]) if ident in type_and_parameter_names]
                construct["private"]       = [ident for ident in index_record.get("private",[]) if ident in type_and_parameter_names]
            construct["used_modules"] = []
            simple_use_statements_module_names = set()
            for used_module1 in index_record["used_modules"]:
                used_module = copy.deepcopy(used_module1)
                if len(used_module1["renamings"]) or len(used_module1["only"]):
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
                    construct["used_modules"].append(used_module)
                else:
                    if used_module["name"] not in simple_use_statements_module_names:
                        simple_use_statements_module_names.add(used_module["name"])
                        construct["used_modules"].append(used_module)
            handle_use_statements_(construct) # take care of duplicates
            # extension
            construct["declarations"] = []
            iv=0
            it=0
            condv = iv < len(construct["variables"])
            condt = it < len(construct["types"])
            condition = condv or condt
            while condition:
                if condv and condt:
                    if construct["variables"][iv]["lineno"] < construct["types"][it]["lineno"]:
                        ivar = construct["variables"][iv]
                        construct["declarations"].append(indexer.types.render_declaration(ivar))
                        iv += 1
                    else:
                        # TODO render types too
                        #itype = copy.deepcopy(construct["variables"][it])
                        #if "device" in itype["attributes"]:
                        #end  
                        it += 1
                elif condv:
                    ivar = construct["variables"][iv]
                    construct["declarations"].append(indexer.types.render_declaration(ivar))
                    iv += 1
                elif condt:
                    # TODO render types too
                    it += 1
                condv = iv < len(construct["variables"])
                condt = it < len(construct["types"])
                condition = condv or condt
            return construct
        
        def handle_use_statements_(icurrent):
            for used_module in indexer.scope.condense_non_only_groups(
                                 indexer.scope.condense_only_groups(
                                   icurrent["used_modules"])):
                is_third_party_module = ("intrinsic" in used_module["attributes"]
                                         or used_module["name"] in indexer.scope.opts.module_ignore_list)
                if not is_third_party_module:
                    imodule_used = next((ientry for ientry in self.index if ientry["name"] == used_module["name"]),None)
                    if imodule_used != None:
                        append_module(create_fortran_construct_record_("module",imodule_used,[used_module["name"]])) 
                    else:
                        msg = "{}no index record found for module '{}'".format(
                            indent,
                            used_module["name"])
                        raise util.error.LookupError(msg)

        scope_tag_tokens = self.scope["tag"].split(":")
        current = None
        for i,_ in enumerate(scope_tag_tokens):
            if i == 0:
                current = next((ientry for ientry in self.index if ientry["name"] == scope_tag_tokens[0]),None)
            else:
                parent = current
                current = next((ientry for ientry in parent["procedures"] if ientry["name"] == scope_tag_tokens[i]),None)
            make_program = i == len(scope_tag_tokens)-1
            kind = "program" if make_program else "module"
            construct = create_fortran_construct_record_(kind,current,scope_tag_tokens[0:i+1])
            parent_module_name = "_".join(scope_tag_tokens[:i])
            # must come afterwards
            if len(parent_module_name):
                construct["used_modules"].insert(0,indexer.create_index_record_from_use_statement("use {}".format(parent_module_name)))
            if make_program:
                construct["used_modules"].insert(0,indexer.create_index_record_from_use_statement("use iso_c_binding"))
                construct["used_modules"].insert(1,indexer.create_index_record_from_use_statement("use iso_fortran_env"))
                program = construct
            else:
                append_module(construct)

        return modules, program

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
