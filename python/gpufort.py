#!/usr/bin/env python3
import os, sys, traceback
import subprocess
import copy, shutil
import argparse
import hashlib
from collections import Iterable # < py38
import pprint    
import logging

import json
import multiprocessing

# local imports
import addtoplevelpath
import utils
import scanner.scanner as scanner
import indexer.indexer as indexer
import translator.translator as translator
import fort2hip.fort2hip as fort2hip

# GLOBAL VARS
LOG_LEVEL = logging.INFO

# arg for kernel generator
# array is split into multiple args

def createIndex(searchDirs,filePath,indexFile):
    if indexFile is not None:
        return json.load(indexFile)
    else: 
        searchDirs.insert(0,os.path.dirname(filePath))
        cleanedSearchDirs = list(set([os.path.abspath(s) for s in searchDirs]))
        context = indexer.scanSearchDirs(cleanedSearchDirs,optionsAsStr="")
        return indexer.resolveDependencies(context,searchedFiles=[ filePath ])

def translateFortranSource(fortranFilePath,stree,index,wrapInIfdef):
    # derive code line groups from original tree
    groups = scanner.groupObjects(stree)
    # first pass to update some scanner tree nodes
    for group in groups.values():
        group.generateCode(index,wrapInIfdef)
    # post process before grouping again
    basename      = os.path.basename(fortranFilePath).split(".")[0]
    hipModuleName = basename.replace(".","_").replace("-","_") + "_kernels"
    scanner.postProcess(stree,hipModuleName)
    
    # group again with updated tree
    groups = scanner.groupObjects(stree)

    # now process file
    outputLines = []
    nextLinesToSkip = 0
    with open(fortranFilePath,"r") as fortranFile:
        lines = scanner.handleIncludeStatements(fortranFilePath,fortranFile.readlines())
        for lineno,line in enumerate(lines):
            if lineno in groups:
                lines, linesToSkip = groups[lineno].generateCode(index,wrapInIfdef)
                nextLinesToSkip = linesToSkip 
                outputLines += lines
            elif nextLinesToSkip > 0:
                nextLinesToSkip -= 1
            else:
                outputLines.append(line)
    parts = os.path.splitext(fortranFilePath)
    modifiedFortranFilePath = "{}.hipified.f90".format(parts[0])
    with open(modifiedFortranFilePath,"w") as outfile:
         for line in outputLines:
              outfile.write(line.strip("\n") + "\n")
    utils.prettifyFFile(modifiedFortranFilePath)
    msg = "created hipified input file:         {}".format(modifiedFortranFilePath)
    logger = logging.getLogger('')
    logger.info(msg) ; print(msg)

def configAddCommandLineArguments(parser):
    """
    The user can add arguments to the command line argument parser here.
    """
    pass

def configAfterCommandLineParsing(args):
    """
    The user can work with parsed command line options here.
    """
    pass

def parseConfig():
    global LOG_LEVEL
    global configAddCommandLineArguments
    global configAfterCommandLineParsing

    # DEFAULTS:
    scanner.GPUFORT_IFDEF       = "__HIP" 
    scanner.CUF_IFDEF           = "CUDA"
    # cublas_v1 routines do not have an handle. cublas v2 routines do
    scanner.CUBLAS_VERSION      = 1
    translator.CUBLAS_VERSION   = 1
    # look for integer (array) with this name; disable: leave empty string or comment out
    translator.HINT_CUFFT_PLAN  = r"cufft_plan\w+"
    # look for integer (array) with this name; disable: leave empty string or comment out
    translator.HINT_CUDA_STREAM = r"stream|strm"
    # logging
    # Use one of 'FATAL','CRITICAL','ERROR','WARNING','WARN','INFO','DEBUG','DEBUG2','DEBUG3','NOTSET' 
    LOG_LEVEL = logging.INFO
    
    # CUF options
    scanner.HIP_MODULE_NAME="hipfort"
    scanner.HIP_MATH_MODULE_PREFIX=scanner.HIP_MODULE_NAME+"_"

    # ACC options
    scanner.ACC_USE_COUNTERS=False
    scanner.ACC_DEV_PREFIX=""
    scanner.ACC_DEV_SUFFIX="_d"
    
    try:
        prolog="""global LOG_LEVEL
global configAddCommandLineArguments
global configAfterCommandLineParsing
"""
        epilog="""
configAddCommandLineArguments = config_add_cl_arguments
configAfterCommandLineParsing = config_after_cl_parsing
"""
        CONFIG_FILE=os.path.dirname(os.path.realpath(__file__))+"/config.py"
        exec(prolog + open(CONFIG_FILE).read()+ epilog)
        msg = "config file '{}' found and successfully parsed".format(CONFIG_FILE)
        logging.getLogger("").info(msg) ; print(msg)
    except FileNotFoundError:
        msg = "no 'config.py' file found" 
        logging.warn(msg)
    except Exception as e:
        msg = "failed to parse config file 'config.py'. REASON: {} (NOTE: prolog lines: {}). ABORT".format(str(e),len(prolog.split("\n")))
        logging.getLogger("").error(msg) 
        sys.exit(1)

def parseCommandLineArguments():
    global LOG_LEVEL
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description="S2S translation tool for CUDA Fortran and Fortran+X")
    
    parser.add_argument("input",help="The input file.",type=argparse.FileType("r"),nargs="?",default=None)
    parser.add_argument("-o,--output", help="The output file. Interface module and HIP C++ implementation are named accordingly", default=sys.stdout, required=False, type=argparse.FileType("w"))
    parser.add_argument("-d,--search-dirs", dest="searchDirs", help="Module search dir", nargs="*",  required=False, default=[], type=str)
    parser.add_argument("-i,--index", dest="index", help="Pregenerated JSON index file. If this option is used, the '-d,--search-dirs' switch is ignored.", required=False, default=None, type=argparse.FileType("r"))
    parser.add_argument("-w,--wrap-in-ifdef",dest="wrapInIfdef",action="store_true",help="Wrap converted lines into ifdef in host code.")
    parser.add_argument("-k,--only-generate-kernels",dest="onlyGenerateKernels",action="store_true",help="Only generate kernels; do not modify host code.")
    parser.add_argument("-m,--only-modify-host-code",dest="onlyModifyHostCode",action="store_true",help="Only modify host code; do not generate kernels.")
    parser.add_argument("-E,--destination-dialect",dest="destinationDialect",default=None,type=str,help="One of: {}".format(", ".join(scanner.SUPPORTED_DESTINATION_DIALECTS)))
    parser.add_argument("--log-level",dest="logLevel",required=False,type=str,default="",help="Set log level. Overrides config value.")
    parser.add_argument("--cublas-v2",dest="cublasV2",action="store_true",help="Assume cublas v2 function signatures that use a handle. Overrides config value.")
    parser.add_argument("--print-config-defaults",dest="printConfigDefaults",action="store_true",help="Print config defaults. "+\
            "Config values can be overriden by providing a config file. A number of config values can be overwritten via this CLI.")
    parser.add_argument("--config-file",type=argparse.FileType("r"),dest="configFile",help="Provide a config file.")
    parser.set_defaults(printConfigDefaults=False,overwriteExisting=True,wrapInIfdef=False,cublasV2=False,onlyGenerateKernels=False,onlyModifyHostCode=False)
    configAddCommandLineArguments(parser)
    args = parser.parse_args()
    configAfterCommandLineParsing(args)

    if args.printConfigDefaults:
        gpufortPythonDir=os.path.dirname(os.path.realpath(__file__))
        optionsFiles = [
          "scanner/scanner_options.py.in",
          "translator/translator_options.py.in",
          "fort2hip/fort2hip_options.py.in",
          "indexer/indexer_options.py.in",
          "indexer/indexertools_options.py.in"
        ]
        print("\nCONFIGURABLE GPUFORT OPTIONS (DEFAULT VALUES):")
        for optionsFile in optionsFiles:
            prefix = optionsFile.split("/")[1].split("_")[0]
            print("\n---- "+prefix+" -----------------------------")
            with open(gpufortPythonDir+"/"+optionsFile) as f:
                for line in f.readlines():
                   if len(line.strip()):
                       if line[0] not in [" ","#","}","]"] and "=" in line:
                           parts = line.split("=")
                           key   = (prefix+"."+parts[0].rstrip()).ljust(36)
                           value = parts[1].lstrip()
                           print(key+"= "+value,end="")
                       else:
                           print(line,end="")
        print("")
        sys.exit(0)
    # mutually exclusive arguments
    if args.onlyGenerateKernels and args.onlyModifyHostCode:
        logging.getLogger("").error("switches '--only-generate-kernels' and 'only-modify-host-code' cannot be used at the same time.") 
        sys.exit()
    # overwrite config values
    if len(args.logLevel):
        LOG_LEVEL = getattr(logging,args.logLevel.upper(),getattr(logging,"INFO"))
    if args.cublasV2:
        scanner.CUBLAS_VERSION = 2
        translator.CUBLAS_VERSION = 2
    # check if input is set
    if args.input is None:
        print("ERROR: no input file",file=sys.stderr)
        sys.exit(2)
    return args

def initLogging(inputFilePath):
    global LOG_LEVEL
    # add custom log levels:
    utils.addLoggingLevel("DEBUG2", logging.DEBUG-1, methodName="debug2")
    utils.addLoggingLevel("DEBUG3", logging.DEBUG-2, methodName="debug3")
    inputFilePathHash = hashlib.md5(inputFilePath.encode()).hexdigest()[0:8]
    logDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../log")
    #shutil.rmtree(logDir,ignore_errors=True)
    os.makedirs(logDir,exist_ok=True)
    
    FORMAT = "gpufort:%(levelname)s:{0}: %(message)s".format(inputFilePath)
    FILE="{0}/log-{1}.log".format(logDir,inputFilePathHash)
    logging.basicConfig(format=FORMAT,filename=FILE,filemode="w", level=LOG_LEVEL)
   
    logger = logging.getLogger('')
    #logger.handlers = []
 
    msg = "input file: {0} (log id: {1})".format(inputFilePath,inputFilePathHash)
    logger.info(msg) ; print(msg)
    msg = "log file:   {0} (log level: {1}) ".format(FILE,logging.getLevelName(LOG_LEVEL))
    logger.info(msg) ; print(msg)

if __name__ == "__main__":
    # read config and command line arguments
    parseConfig()
    args = parseCommandLineArguments()
    inputFilePath = os.path.abspath(args.input.name)
    
    # init logging
    initLogging(inputFilePath)

    # parse file and create index in parallel
    if args.destinationDialect is not None:
        scanner.DESTINATION_DIALECT = \
          scanner.checkDestinationDialect(args.destinationDialect)
    stree = None
    index = []
    with multiprocessing.Pool(processes=2) as pool: 
        handle = pool.apply_async(\
           scanner.parseFile, (inputFilePath,))
        index = createIndex(args.searchDirs,inputFilePath,args.index)
        pool.close()
        pool.join()
        stree = handle.get()
   
    # extract kernels
    outputFilePrefix = ".".join(inputFilePath.split(".")[:-1])
    basename         =  os.path.basename(outputFilePrefix)
    if "hip" in scanner.DESTINATION_DIALECT: 
        kernelsToConvertToHip = ["*"]
    else:
        kernelsToConvertToHip = scanner.KERNELS_TO_CONVERT_TO_HIP
    fort2hip.createHipKernels(stree,index,kernelsToConvertToHip,outputFilePrefix,basename,\
            generateCode=not args.onlyModifyHostCode)
    
    # modify original file
    if not args.onlyGenerateKernels:
        translateFortranSource(inputFilePath,stree,index,args.wrapInIfdef) 
 
    # shutdown logging
    logging.shutdown()
