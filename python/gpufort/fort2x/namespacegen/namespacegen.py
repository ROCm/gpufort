# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import copy
import tempfile

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

    def __init__(self,scope,**kwargs):
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
        self.scope = scope
        self.cpp_lines_to_prepend,_ = util.kwargs.get_value("cpp_lines_to_prepend", opts.prepend_callback(scope["tag"]), **kwargs)
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

    def __resolve_all_parameters_via_compiler(self):
        already_considered = set()
        decl_list = []
        ivars_to_resolve = []
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
                decl_list.insert(0,indexer.types.render_declaration(ivar)) # preprend as we have reversed
                # if compiled and excecuted prints the following:
                ivars_to_resolve.insert(0,ivar)
        #print(decl_list)
        #print(ivars_to_resolve)
        fortran_snippet = render.render_resolve_scope_program_f03(decl_list,ivars_to_resolve)
        #print(fortran_snippet)
        temp_infile = tempfile.NamedTemporaryFile(
            delete=False, 
            mode="w",
            prefix="gpufort-namespacegen",
            suffix=".f90")
        temp_infile.write(fortran_snippet)
        temp_infile.close()
        temp_outfile_path = temp_infile.name.replace(".f90",".x")
        cmd_compile = [self.fortran_compiler,self.fortran_compiler_flags,temp_infile.name,"-o",temp_outfile_path]
        status,_,err_out = util.subprocess.run_subprocess(cmd_compile,True)
        if status != 0:
            raise util.error.LookupError("failed resolving parameters in scope '{}' as compilation with compiler '{}' and flags '{}' failed for the following reason: {}".format(
                self.scope["tag"],self.fortran_compiler," ".join(self.fortran_compiler_flags),err_out))
        if os.path.exists(temp_infile.name):
            os.remove(temp_infile.name)
        cmd_run     = [temp_outfile_path]
        _,std_out,_ = util.subprocess.run_subprocess(cmd_run,True)
        if os.path.exists(temp_outfile_path):
            os.remove(temp_outfile_path)
        # now fill the namespace body with the resolved parameters
        body = []
        for line in std_out.splitlines():
            #print(line)
            result,_ =\
                util.parsing.get_top_level_operands(util.parsing.tokenize(line)) 
            #print(result)
            name, f_type, f_len, kind, bpe, rank, sizes, lbounds, rhs_expr = result
            if f_len == "<none>":
                f_len == None
            ivar = indexer.types.create_index_var(f_type,f_len,kind,[],name,[],[],rhs_expr)
            translator.analysis.append_c_type(ivar)
            tokens = [" "*2,"constexpr ",ivar["c_type"]]
            tokens += [" ",ivar["name"]," = ",rhs_expr]
            if ivar["c_type"] == "float":
                tokens.append("f")
            tokens.append(";")
            body.append("".join(tokens))
        return body 

    def render_namespace_cpp(self):
        scope_tag = self.scope["tag"]
        result = [ "namespace _{} {{".format(scope_tag.replace(":","_"))]
        if self.resolve_all_parameters_via_compiler:
            result += self.__resolve_all_parameters_via_compiler()  
        else:
            result += self.__convert_rhs_to_cpp()
        result.append("}")
        return result
