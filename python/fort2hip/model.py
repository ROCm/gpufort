# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

import addtoplevelpath
import utils.logging
    
class BaseModel():
    def __init__(self,template):
        self._template = template
    def generate_code(self,context={}):
        LOADER = jinja2.FileSystemLoader(os.path.realpath(os.path.dirname(__file__)))
        ENV    = jinja2.Environment(loader=LOADER, trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)

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
        BaseModel.__init__(self,"templates/hip_implementation.template.cpp")

class InterfaceModuleModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/interface_module.template.f03")

class InterfaceModuleTestModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/interface_module_test.template.f03")

class GpufortHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/gpufort.template.h")

class GpufortReductionsHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/gpufort_reductions.template.h")

class GpufortArraysHeaderModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/gpufort_arrays.template.h")

class GpufortArraysFortranInterfacesModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self,"templates/gpufort_arrays.template.f03")

#model = GpufortHeaderModel()
#model.generate_file("gpufort.h")
#model = GpufortReductionsHeaderModel()
#model.generate_file("gpufort_reductions.h")

#model_c = GpufortArraysModel().generate_file("gpufort_arrays.h",
#  context={"max_rank":7})
#model_f = GpufortArraysFortranInterfaceModel().generate_file("gpufort_arrays.f90")
