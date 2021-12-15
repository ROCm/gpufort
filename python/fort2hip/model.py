# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

import addtoplevelpath
import utils.logging
        
LOADER    = jinja2.FileSystemLoader(os.path.realpath(os.path.join(os.path.dirname(__file__),"templates")))
ENV       = jinja2.Environment(loader=LOADER, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
TEMPLATES = {}
    
def generate_code(template_path,context={}):
    global ENV
    global TEMPLATES
    if template_path in TEMPLATES:
        template = TEMPLATES[template_path]
    else:
        template = ENV.get_template(template_path)
        TEMPLATES[template_path] = template
    try:
        return template.render(context)
    except Exception as e:
        utils.logging.log_error("fort2hip.model","generate_code","could not render template '%s'" % template_path)
        raise e
    
class BaseModel():
    def __init__(self,template):
        self._template = template
    def generate_code(self,context={}):
        global ENV 
        template = ENV.get_template(self._template)
        try:
            return template.render(context)
        except Exception as e:
            utils.logging.log_error("fort2hip.model","BaseModel.generate_code","could not render template '%s'" % self._template)
            raise e
    def generate_file(self,output_file_path,context={}):
        with open(output_file_path, "w") as output:
            output.write(self.generate_code(context))

class HipImplementationModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"hip_implementation.template.hip.cpp")

class InterfaceModuleModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"interface_module.template.f03")

class HipImplementationLegacyModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"legacy/hip_implementation.template.cpp")

class InterfaceModuleLegacyModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"legacy/interface_module.template.f03")

#class InterfaceModuleTestModel(BaseModel):
#    def __init__(self):
#        BaseModel.__init__(self,"interface_module_test.template.f03")

class GpufortHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"gpufort.template.h")

class GpufortReductionHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"gpufort_reduction.template.h")

class GpufortArrayHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"gpufort_array.template.h")

class GpufortArraySourceModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"gpufort_array.template.hip.cpp")

class GpufortArrayFortranInterfacesModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"gpufort_array.template.f03")

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"templates","model.py.in")).read())

#model = GpufortHeaderModel()
#model.generate_file("gpufort.h")
#model = GpufortReductionHeaderModel()
#model.generate_file("gpufort_reduction.h")

#model_c = GpufortArrayModel().generate_file("gpufort_array.h",
#  context={"max_rank":7})
#model_f = GpufortArrayFortranInterfaceModel().generate_file("gpufort_array.f90")
