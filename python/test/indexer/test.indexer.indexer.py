#!/usr/bin/env python3
import addtoplevelpath
import indexer.indexer as indexer
import indexer.scoper as scoper
import utils.logging

import json

utils.logging.VERBOSE   = True
utils.logging.initLogging("log.log","debug3")

gfortranOptions="-DCUDA"

# dependency
writtenIndex = []
indexer.scanFile("test_modules.f90",gfortranOptions,writtenIndex)
indexer.writeGpufortModuleFiles(writtenIndex,"./")

# main file
writtenIndex.clear()
indexer.scanFile("test1.f90",gfortranOptions,writtenIndex)
indexer.writeGpufortModuleFiles(writtenIndex,"./")

# read
readIndex = []
indexer.loadGpufortModuleFiles(["./"],readIndex)

#print(json.dumps(readIndex,indent=2))
