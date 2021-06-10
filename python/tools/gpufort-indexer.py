# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import re
import logging
import json
import time

import indexer.indexer as indexer
import utils.logging
    
def parseCommandLineArguments():
    args = {}
    args["options"] = sys.argv[1:]

    if not len(args["options"]) or "-h" in args["options"] or "--help" in args["options"]:
        print("HIPFORTIFY-MODSCAN: description='Discover and represent Fortran modules and their content as JSON object")
        print("\nusage: ./gpufort-indexer.py [gfortran options]")
        print("\nexample usage: ./gpufort-indexer.py -I/usr/local/include -DUSE_A_FEATURE")
        sys.exit(0)
    args["searchDirs"] = [ opt[2:] for opt in args["options"] if opt.startswith("-I") ]
    args["logLevel"] = "DEBUG2"
    
    # filtering
    args["searchedFiles"] = []
    args["searchedTags"]  = []
    current = None
    end = -1
    for i,opt in enumerate(args["options"]):
         if opt == "--files":
             if end < 0:
                 end = i
             current = args["searchedFiles"] 
         if opt == "--tags":
             if end < 0:
                 end = i
             current = args["searchedTags"]
         if current != None:
             current.append(opt)
    if end > 0:
        args["options"] = sys.argv[1:end]
    return args

def initLogging(args):
    # add custom log levels:
    utils.logging.addAdditionalDebugLevels()
    logDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"log")
    os.makedirs(logDir,exist_ok=True)
   
    LOG_LEVEL = getattr(logging,args["logLevel"],getattr(logging,"INFO"))
    FORMAT = "gpufort-indexer:%(levelname)s: %(message)s"
    FILE="{0}/log.log".format(logDir)
    logging.basicConfig(format=FORMAT,filename=FILE,filemode="w", level=LOG_LEVEL)
   
    msg = "log file:   {0} (log level: {1}) ".format(FILE,logging.getLevelName(LOG_LEVEL))
    utils.logging.logInfo(msg)

if __name__ == "__main__":
    start_time = time.time()

    args = parseCommandLineArguments()
    initLogging(args)
    
    begin_timing = time.time()
    optionsAsStr = " ".join(args["options"])
    index = indexer.scanSearchDirs(args["searchDirs"],optionsAsStr,prescan=False)
    print("scan files:            %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    #
    begin_timing = time.time()
    index = indexer.resolveDependencies(index,args["searchedFiles"],args["searchedTags"])
    print("resolve dependencies: %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    #
    indexer.PRETTY_PRINT_INDEX_FILE = True
    begin_timing = time.time()
    filepath = "index.json"
    indexer.writeIndexToFile(index,filepath)
    print("save index:          %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    begin_timing = time.time()
    _ = indexer.loadIndexFromFile(filepath)
    print("load saved index:    %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    
    print("--- %s seconds ---" % (time.time() - start_time),file=sys.stderr)
