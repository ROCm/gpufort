# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

import addtoplevelpath
import utils.logging
    
class BaseModel():
    def __init__(self,template):
        self._template = template
    def generateCode(self,context={}):
        LOADER = jinja2.FileSystemLoader(os.path.realpath(os.path.dirname(__file__)))
        ENV    = jinja2.Environment(loader=LOADER, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)

        template = ENV.get_template(self._template)
        try:
            return template.render(context)
        except Exception as e:
            utils.logging.logError("fort2hip.model","BaseModel.generateCode","could not render template '%s'" % self._template)
            raise e
    def generateFile(self,outputFilePath,context={}):
        with open(absolutePath, "w") as output:
            output.write(generateCode(context))
        return absolutePath, context

class HipImplementationModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/HipImplementation.template.cpp")

class InterfaceModuleModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/InterfaceModule.template.f03")

class InterfaceModuleTestModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/InterfaceModuleTest.template.f03")

class GpufortHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/Gpufort.template.h")

class GpufortReductionsHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/GpufortReductions.template.h")

#model = GpufortHeaderModel()
#model.generateFile("gpufort.h")
#model = GpufortReductionsHeaderModel()
#model.generateFile("gpufort_reductions.h")
