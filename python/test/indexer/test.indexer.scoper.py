#!/usr/bin/env python3
import addtoplevelpath
import indexer.indexer as indexer
import translator.translator as translator
import indexer.scoper as scoper
import utils.logging

utils.logging.VERBOSE    = True
utils.logging.initLogging("log.log","debug3")

gfortranOptions="-DCUDA"

# dependency
index = []
indexer.scanFile("test_modules.f90",gfortranOptions,index)

# main file
indexer.scanFile("test1.f90",gfortranOptions,index)

parentTag = "nested_subprograms"
c   = scoper.searchIndexForVariable(index,"test1","c") # included from module 'simple'
t_b = scoper.searchIndexForVariable(index,"test1",\
  translator.createIndexSearchTagForVariable("t%b")) # type of t included from module 'simple'

tc_t1list_a = scoper.searchIndexForVariable(index,"test1",\
  translator.createIndexSearchTagForVariable("tc%t1list(i)%a")) # type of t included from module 'simple'
tc_t2list_t1list_a = scoper.searchIndexForVariable(index,"test1",
  translator.createIndexSearchTagForVariable("tc%t2list(indexlist%j)%t1list(i)%a")) # type of t included from module 'simple'

#print(json.dumps(scope,indent=2))
#c = scoper.searchIndexForVariable(index,"test1","func2")

#print(json.dumps(scope,indent=2))
