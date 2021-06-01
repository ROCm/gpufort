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
import utils
    
PRETTY_PRINT_INDEX = False

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
    utils.addLoggingLevel("DEBUG2", logging.DEBUG-1, methodName="debug2")
    utils.addLoggingLevel("DEBUG3", logging.DEBUG-2, methodName="debug3")
    logDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"log")
    os.makedirs(logDir,exist_ok=True)
   
    LOG_LEVEL = getattr(logging,args["logLevel"],getattr(logging,"INFO"))
    FORMAT = "gpufort-indexer:%(levelname)s: %(message)s"
    FILE="{0}/log.log".format(logDir)
    logging.basicConfig(format=FORMAT,filename=FILE,filemode="w", level=LOG_LEVEL)
   
    msg = "log file:   {0} (log level: {1}) ".format(FILE,logging.getLevelName(LOG_LEVEL))
    utils.logInfo(msg)

if __name__ == "__main__":
    start_time = time.time()

    args = parseCommandLineArguments()
    initLogging(args)
    
    begin_timing = time.time()
    optionsAsStr = " ".join(args["options"])
    context = indexer.scanSearchDirs(args["searchDirs"],optionsAsStr,prescan=False)
    print("scan files:            %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    #
    begin_timing = time.time()
    context = indexer.resolveDependencies(context,args["searchedFiles"],args["searchedTags"])
    print("resolve dependencies: %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    #
    begin_timing = time.time()
    with open("index.json","w") as outfile:
        if PRETTY_PRINT_INDEX:
            json.dump(context,outfile,indent=4,sort_keys=False)
        else:
            json.dump(context,outfile)
    print("save context:          %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    begin_timing = time.time()
    with open("index.json","r") as infile:
        data = json.load(infile)
    print("load saved context:    %s seconds" % (time.time() - begin_timing),file=sys.stderr)
    
    print("--- %s seconds ---" % (time.time() - start_time),file=sys.stderr)
