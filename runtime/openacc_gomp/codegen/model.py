#!/usr/bin/env python3
##
# @section DESCRIPTION
#
# Generates an interface model for the gpufort acc runtime,
# which is based on hipfort's Fortran interfaces for HIP.
#

from abstractModelBaseClass import AbstractModelBaseClass


class Model(AbstractModelBaseClass):

    def generateCode(self,outputFilePath):
        return self.render("templates/openacc_gomp.f-template", outputFilePath) #return path to generated file

if __name__ == "__main__":
    maxDims = 7

    mappings = [
            ["present","GOMP_MAP_FORCE_PRESENT"],
            ["create","GOMP_MAP_ALLOC"],
            ["no_create","GOMP_MAP_ALLOC"],
            ["copy","GOMP_MAP_TOFROM"],
            ["copyin","GOMP_MAP_TO"],
            ["copyout","GOMP_MAP_FROM"],
            ["delete","GOMP_MAP_DELETE"]
    ]

    datatypes  =  [\
            ["l","1","logical"], \
            ["i4", "4", "integer(4)"] ,["i8","8","integer(8)"], \
            ["r4","4","real(4)"], ["r8","8","real(8)"], \
            ["c4","2*4","complex(4)"],["c8","2*8","complex(8)"] \
    ]
    dimensions = range(0,maxDims+1)
    context = { 
            "mappings"   : mappings, 
            "datatypes"  : datatypes,
            "datatypes"  : datatypes,
            "dimensions" : dimensions }
    
    model = Model(context)
    model.generateCode("openacc_gomp.f90")
