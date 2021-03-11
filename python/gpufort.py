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

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpufort_options.py.in")).read())

# arg for kernel generator
# array is split into multiple args

def createIndex(searchDirs,defines,filePath,indexFile):
    if indexFile is not None:
        return json.load(indexFile)
    else: 
        searchDirs.insert(0,os.path.dirname(filePath))
        context = indexer.scanSearchDirs(searchDirs,optionsAsStr=" ".join(defines))
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
    msg = "created hipified input file: ".ljust(40) + modifiedFortranFilePath
    logger = logging.getLogger("")
    logger.info(msg) ; print(msg)

def parseRawCommandLineArguments():
    """
    Parse command line arguments before using argparse.
    Allows the loaded config file to modify
    the argparse switches and help information.
    Further transform some arguments.
    """
    configFilePath = None
    workingDirPath = os.getcwd()
    includeDirs    = []
    defines        = []
    options = sys.argv[1:]
    for i,opt in enumerate(list(options)):
        if opt == "--working-dir":
            if i+1 < len(options):
                workingDirPath = options[i+1]
        elif opt == "--config-file":
            if i+1 < len(options):
                configFilePath = options[i+1]
        elif opt.startswith("-I"):
            includeDirs.append(opt[2:])
            sys.argv.remove(opt)
        elif opt.startswith("-D"):
            defines.append(opt)
            sys.argv.remove(opt)
    if not os.path.exists(workingDirPath):
        msg = "working directory '{}' cannot be found".format(workingDirPath)
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    if configFilePath is not None: 
        if configFilePath[0] != "/":
          configFilePath = workingDirPath + "/" + configFilePath 
        if not os.path.exists(configFilePath):
            msg = "config file '{}' cannot be found".format(configFilePath)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(2)
    return configFilePath, includeDirs, defines 

def parseConfig(configFilePath):
    """
    Load user-supplied config fule.
    """
    global LOG_LEVEL
    global LOG_DIR
    global LOG_DIR_CREATE

    prolog = "global LOG_LEVEL\nglobal LOG_DIR\nglobal LOG_DIR_CREATE"
    epilog = ""
    try:
        if configFilePath.strip()[0]=="/":
            CONFIG_FILE = configFilePath
        else:
            CONFIG_FILE=os.path.dirname(os.path.realpath(__file__))+"/"+configFilePath
        exec(prolog + open(CONFIG_FILE).read()+ epilog)
        msg = "config file '{}' found and successfully parsed".format(CONFIG_FILE)
        logging.getLogger("").info(msg) ; print(msg)
    except FileNotFoundError:
        msg = "no '{}' file found. Use defaults.".format(CONFIG_FILE)
        logging.warn(msg)
    except Exception as e:
        msg = "failed to parse config file '{}'. REASON: {} (NOTE: num prolog lines: {}). ABORT".format(CONFIG_FILE,str(e),len(prolog.split("\n")))
        logging.getLogger("").error(msg) 
        sys.exit(1)

def parseCommandLineArguments():
    """
    Parse command line arguments after all changes and argument transformations by the config file have been applied.
    """
    global LOG_LEVEL
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description="S2S translation tool for CUDA Fortran and Fortran+X")
    
    parser.add_argument("input",help="The input file.",type=str,nargs="?",default=None)
    parser.add_argument("-o,--output", help="The output file. Interface module and HIP C++ implementation are named accordingly", default=sys.stdout, required=False, type=argparse.FileType("w"))
    parser.add_argument("-d,--search-dirs", dest="searchDirs", help="Module search dir", nargs="*",  required=False, default=[], type=str)
    parser.add_argument("-i,--index", dest="index", help="Pregenerated JSON index file. If this option is used, the '-d,--search-dirs' switch is ignored.", required=False, default=None, type=argparse.FileType("r"))
    parser.add_argument("-w,--wrap-in-ifdef",dest="wrapInIfdef",action="store_true",help="Wrap converted lines into ifdef in host code.")
    parser.add_argument("-k,--only-generate-kernels",dest="onlyGenerateKernels",action="store_true",help="Only generate kernels; do not modify host code.")
    parser.add_argument("-m,--only-modify-host-code",dest="onlyModifyHostCode",action="store_true",help="Only modify host code; do not generate kernels.")
    parser.add_argument("-E,--destination-dialect",dest="destinationDialect",default=None,type=str,help="One of: {}".format(", ".join(scanner.SUPPORTED_DESTINATION_DIALECTS)))
    parser.add_argument("--log-level",dest="logLevel",required=False,type=str,default="",help="Set log level. Overrides config value.")
    parser.add_argument("--cublas-v2",dest="cublasV2",action="store_true",help="Assume cublas v2 function signatures that use a handle. Overrides config value.")
    # shadow arguments that are actually taken care of by raw argument parsing
    parser.add_argument("--working-dir",dest="workingDir",default=os.getcwd(),type=str,help="Set working directory.")
    parser.add_argument("--print-config-defaults",dest="printConfigDefaults",action="store_true",help="Print config defaults. "+\
            "Config values can be overriden by providing a config file. A number of config values can be overwritten via this CLI.")
    parser.add_argument("--config-file",default=None,type=argparse.FileType("r"),dest="configFile",help="Provide a config file.")
    
    parser.set_defaults(printConfigDefaults=False,overwriteExisting=True,wrapInIfdef=False,cublasV2=False,onlyGenerateKernels=False,onlyModifyHostCode=False)
    args, unknownArgs = parser.parse_known_args()

    if args.printConfigDefaults:
        gpufortPythonDir=os.path.dirname(os.path.realpath(__file__))
        optionsFiles = [
          "./gpufort_options.py.in",
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
                           if prefix == "gpufort":
                               key   = (parts[0].rstrip()).ljust(36)
                           else:
                               key   = (prefix+"."+parts[0].rstrip()).ljust(36)
                           value = parts[1].lstrip()
                           print(key+"= "+value,end="")
                       else:
                           print(line,end="")
        print("")
        sys.exit(0)
    # mutually exclusive arguments
    if args.onlyGenerateKernels and args.onlyModifyHostCode:
        msg = "switches '--only-generate-kernels' and 'only-modify-host-code' cannot be used at the same time."
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    # overwrite config values
    if len(args.logLevel):
        LOG_LEVEL = getattr(logging,args.logLevel.upper(),getattr(logging,"INFO"))
    if args.cublasV2:
        scanner.CUBLAS_VERSION = 2
        translator.CUBLAS_VERSION = 2
    # check if input is set
    if args.input is None:
        msg = "no input file"
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    if args.input[0] != "/":
        args.input = args.workingDir + "/" + args.input 
    if not os.path.exists(args.input):
        msg = "input file '{}' cannot be found".format(args.input)
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    if len(unknownArgs):
        msg = "unknown arguments (may be used by registered actions): {}".format(" ".join(unknownArgs))
        print("WARNING: "+msg,file=sys.stderr)
    return args, unknownArgs

def initLogging(inputFilePath):
    global LOG_LEVEL
    global LOG_DIR
    global LOG_DIR_CREATE
    # add custom log levels:
    utils.addLoggingLevel("DEBUG2", logging.DEBUG-1, methodName="debug2")
    utils.addLoggingLevel("DEBUG3", logging.DEBUG-2, methodName="debug3")
    inputFilePathHash = hashlib.md5(inputFilePath.encode()).hexdigest()[0:8]
    logDir = LOG_DIR
    if not LOG_DIR_CREATE and not os.path.exists(logDir):
        msg = "directory for storing log files ('{}') does not exist".format(logDir)
        print("ERROR: "+msg)
        sys.exit(2)
    os.makedirs(logDir,exist_ok=True)
    
    FORMAT = "gpufort:%(levelname)s:{0}: %(message)s".format(inputFilePath)
    FILE="{0}/log-{1}.log".format(logDir,inputFilePathHash)
    logging.basicConfig(format=FORMAT,filename=FILE,filemode="w", level=LOG_LEVEL)
   
    logger = logging.getLogger("")
    msg = "input file: {0} (log id: {1})".format(inputFilePath,inputFilePathHash)
    logger.info(msg) ; print(msg)
    msg = "log file:   {0} (log level: {1}) ".format(FILE,logging.getLevelName(LOG_LEVEL))
    logger.info(msg) ; print(msg)

if __name__ == "__main__":
    # read config and command line arguments
    configFilePath, includeDirs, defines = parseRawCommandLineArguments()
    if configFilePath is not None:
        parseConfig(configFilePath)
    args, unknownArgs = parseCommandLineArguments()
    if len(POST_CLI_ACTIONS):
        msg = "run registered actions"
        logging.getLogger("").info(msg)# ; print(msg,file=sys.stderr)
        for action in POST_CLI_ACTIONS:
            if callable(action):
                action(args,unknownArgs)
    
    inputFilePath = os.path.abspath(args.input)
    args.searchDirs += includeDirs
    oneOrMoreSearchDirsNotFound = False
    for i,directory in enumerate(args.searchDirs):
        if directory[0]  != "/":
            args.searchDirs[i] = args.workingDir+"/"+directory
        if not os.path.exists(args.searchDirs[i]):
            msg = "search directory '{}' cannot be found".format(args.searchDirs)
            print("ERROR: "+msg,file=sys.stderr)
            oneOrMoreSearchDirsNotFound = True
    if oneOrMoreSearchDirsNotFound:
        sys.exit(2)
    
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
        index = createIndex(args.searchDirs,defines,inputFilePath,args.index)
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
