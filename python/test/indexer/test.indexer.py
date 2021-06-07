#!/usr/bin/env python3
import addtoplevelpath
import indexer.indexer as indexer
import indexer.indexertools as indexertools

options="-DCUDA"

# dependency
writtenIndex = []
indexer.scanFile("test_modules.f90","",writtenIndex)
indexer.writeGpufortModuleFiles(writtenIndex,"./")

# main file
writtenIndex.clear()
indexer.scanFile("test1.f90","",writtenIndex)
indexer.writeGpufortModuleFiles(writtenIndex,"./")

# read
readIndex = []
indexer.loadGpufortModuleFiles(["./"],readIndex)

print(readIndex)

#indexer.loadUsedModules(["./"],readIndex)
