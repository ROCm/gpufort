#!/usr/bin/env python3
import addtoplevelpath
import indexer.indexer as indexer
import indexer.scoper as scoper
import utils 

import json

utils.registerAdditionalDebugLevels()

gfortranOptions="-DCUDA"

# dependency
index = []
indexer.scanFile("test_modules.f90",gfortranOptions,index)

# main file
indexer.scanFile("test1.f90",gfortranOptions,index)

parentTag = "mymod2"

scope = scoper.constructScope(index,parentTag,errorHandling="strict")
