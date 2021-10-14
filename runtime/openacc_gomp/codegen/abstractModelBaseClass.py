##
# @file This file is part of the ExaHyPE project.
# @author ExaHyPE Group (exahype@lists.lrz.de)
#
# @section LICENSE
#
# Copyright (c) 2016  http://exahype.eu
# All rights reserved.
#
# The project has received funding from the European Union's Horizon 
# 2020 research and innovation programme under grant agreement
# No 671698. For copyrights and licensing, please consult the webpage.
#
# Released under the BSD 3 Open Source License.
# For the full license text, see LICENSE.txt
#
#
# @section DESCRIPTION
#
# Abstract base class for the Models
#
# provide a render method and the signature for generateCode
#
# copy the provided baseContext as local context (need to hard copy as a model can modify its local context)
#


import copy
import sys
import os
import pprint

## add path to dependencies
#from ..configuration import Configuration
#sys.path.insert(1, Configuration.pathToJinja2)
#sys.path.insert(1, Configuration.pathToMarkupsafe)

import jinja2


class AbstractModelBaseClass():
    """Base class of a Models
    
    Render and __init__ shouldn't need to be overriden
    
    Override generateCode to implement your model. 
    """

    def __init__(self, baseContext):
        self.context = copy.copy(baseContext)
    
    
    def generateCode(self):
        """To be overriden
        
        Generate the code by filling self.context and calling 
        self.render(templatePath, outputFileName)
        """
        sys.exit("Abstract method") # needs to be overriden
    
    
    # render a template to outputFilename using the given context (default = local context)
    def render(self, templateName, outputFilePath, context=None, overwrite=True):
        """Render a template to outputFilename using the given context (local context if none given)
        
        Return the path to the generated file or `None` if the file exists and
        shall not be overwritten and the context
        """
        # set default context to local context if none given
        if context == None:
            context = self.context
        
        absolutePath = outputFilePath
        canonicalPath = os.path.realpath(absolutePath)
        
        if not os.path.exists(canonicalPath) or overwrite:
            with open(canonicalPath, "w") as output:
                output.write(self.renderAsString(templateName, context))
            return canonicalPath, context
        else:
            return None, context
    
    
    def renderAsString(self, templateName, context=None):
        """Return a template rendered as a String using the given context (local context if none given)"""
        # set default context to local context if none given
        if context == None:
            context = self.context
        #loader = codegen.FileSystemLoader(os.path.realpath(os.path.join(os.path.dirname(__file__),"..","templates")))
        loader = jinja2.FileSystemLoader(os.path.realpath(os.path.dirname(__file__)))
        env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
        template = env.get_template(templateName)
        
        try:
            return template.render(context)
        except Exception:
            pp = pprint.PrettyPrinter(depth=8)
            print("ERROR: could not render template '%s'" % templateName,file=sys.stderr)
            print("ERROR: used context:",file=sys.stderr)
            print(pp.pformat(context),file=sys.stderr)
            raise