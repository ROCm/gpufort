# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint
import logging

import jinja2
    
LOADER = jinja2.FileSystemLoader(os.path.realpath(os.path.dirname(__file__)))
ENV    = jinja2.Environment(loader=LOADER, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)

def renderAsString(templateName, context):
    """Return a template rendered as a String using the given context"""
    template = ENV.get_template(templateName)
    
    try:
        return template.render(context)
    except Exception:
        pp = pprint.PrettyPrinter(depth=8)
        logging.logError("could not render template '%s'" % templateName)
        logging.logError("used context:")
        logging.logError(pp.pformat(context))
        raise

def render(templateName, outputFilePath, context):
    """Render a template to outputFilename using the given context.
    Always overwrite.
    """
    absolutePath  = outputFilePath
    canonicalPath = os.path.realpath(absolutePath)
    
    with open(canonicalPath, "w") as output:
        output.write(renderAsString(templateName, context))
    return canonicalPath, context

class HipImplementationModel():
    def generateCode(self,outputFilePath,context):
        return render("templates/HipImplementation.template.cpp", outputFilePath, context)

class InterfaceModuleModel():
    def generateCode(self,outputFilePath,context):
        return render("templates/InterfaceModule.template.f03", outputFilePath, context)

class InterfaceModuleTestModel():
    def generateCode(self,outputFilePath,context):
        return render("templates/InterfaceModuleTest.template.f03", outputFilePath, context) 

class GpufortHeaderModel():
    def generateCode(self,outputFilePath,context={}):
        return render("templates/Gpufort.template.h", outputFilePath, context) 

class GpufortReductionsHeaderModel():
    def generateCode(self,outputFilePath,context={}):
        return render("templates/GpufortReductions.template.h", outputFilePath, context) 

#model = GpufortHeaderModel()
#model.generateCode("gpufort.h")
#model = GpufortReductionsHeaderModel()
#model.generateCode("gpufort_reductions.h")
