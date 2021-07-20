#!/usr/bin/env python3
# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os, sys
import argparse
import hashlib
import cProfile,pstats,io

# local imports
import addtoplevelpath
import utils.logging
import utils.fileutils
import scanner.scanner as scanner
import indexer.indexer as indexer
import linemapper.linemapper as linemapper
import translator.translator as translator
import fort2hip.fort2hip as fort2hip

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpufort_options.py.in")).read())

# arg for kernel generator
# array is split into multiple args

def createIndex(searchDirs,options,filepath):
    global LOG_PREFIX
    global SKIP_CREATE_GPUFORT_MODULE_FILES
   
    utils.logging.logEnterFunction(LOG_PREFIX,"createIndex",\
      {"filepath":filepath,"options":" ".join(options),\
       "searchDirs":" ".join(searchDirs)})

    optionsAsStr = " ".join(options)
    
    index = []
    if not SKIP_CREATE_GPUFORT_MODULE_FILES:
        indexer.scanFile(filepath,optionsAsStr,index)
        outputDir = os.path.dirname(filepath)
        indexer.writeGpufortModuleFiles(index,outputDir)
    index.clear()
    indexer.loadGpufortModuleFiles(searchDirs,index)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"createIndex")
    return index

def __translateSource(infilepath,stree,records,index):
    global LOG_PREFIX
    global PRETTIFY_MODIFIED_TRANSLATION_SOURCE
    global SKIP_CREATE_GPUFORT_MODULE_FILES
    
    utils.logging.logEnterFunction(LOG_PREFIX,"__translateSource",{"infilepath":infilepath})
    
    # post process
    basename      = os.path.basename(infilepath).split(".")[0]
    hipModuleName = basename.replace(".","_").replace("-","_") + "_kernels"
    scanner.postprocess(stree,hipModuleName,index)
    
    # transform statements; wrtites to 'records'
    def transform_(stnode):
        stnode.transformStatements(index)
        for child in stnode._children:
            transform_(child)
    transform_(stree)

    # write the file
    parts = os.path.splitext(infilepath)
    outfilepath = "{}.hipified.f90".format(parts[0])
    linemapper.writeModifiedFile(outfilepath,infilepath,records)

    # prettify the file
    if PRETTIFY_MODIFIED_TRANSLATION_SOURCE:
        utils.fileutils.prettifyFFile(outfilepath)
    msg = "created hipified input file: ".ljust(40) + outfilepath
    utils.logging.logInfo(LOG_PREFIX,"__translateSource",msg)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"__translateSource")

def parseRawCommandLineArguments():
    """
    Parse command line arguments before using argparse.
    Allows the loaded config file to modify
    the argparse switches and help information.
    Further transform some arguments.
    """
    configFilepath = None
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
                configFilepath = options[i+1]
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
    if configFilepath != None: 
        if configFilepath[0] != "/":
          configFilepath = workingDirPath + "/" + configFilepath 
        if not os.path.exists(configFilepath):
            msg = "config file '{}' cannot be found".format(configFilepath)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(2)
    return configFilepath, includeDirs, defines 

def parseConfig(configFilepath):
    """
    Load user-supplied config fule.
    """
    prolog = """
global LOG_LEVEL
global LOG_FORMAT_PATTERN
global ONLY_CREATE_GPUFORT_MODULE_FILES
global SKIP_CREATE_GPUFORT_MODULE_FILES
global ONLY_MODIFY_TRANSLATION_SOURCE
global ONLY_GENERATE_KERNELS
global POST_CLI_ACTIONS
global PRETTIFY_MODIFIED_TRANSLATION_SOURCE
global INCLUDE_DIRS
""".strip()
    epilog = ""
    try:
        if configFilepath.strip()[0]=="/":
            CONFIG_FILE = configFilepath
        else:
            CONFIG_FILE=os.path.dirname(os.path.realpath(__file__))+"/"+configFilepath
        exec(prolog + "\n" + open(CONFIG_FILE).read()+ epilog)
        msg = "config file '{}' found and successfully parsed".format(CONFIG_FILE)
        utils.logging.logInfo(LOG_PREFIX,"parseConfig",msg)
    except FileNotFoundError:
        msg = "no '{}' file found. Use defaults.".format(CONFIG_FILE)
        utils.logging.logWarning(LOG_PREFIX,"parseConfig",msg)
    except Exception as e:
        msg = "failed to parse config file '{}'. REASON: {} (NOTE: num prolog lines: {}). ABORT".format(CONFIG_FILE,str(e),len(prolog.split("\n")))
        utils.logging.logError(LOG_PREFIX,"parseConfig",msg)
        sys.exit(1)

def parseCommandLineArguments():
    """
    Parse command line arguments after all changes and argument transformations by the config file have been applied.
    """
    global LOG_LEVEL
    global SKIP_CREATE_GPUFORT_MODULE_FILES
    global ONLY_CREATE_GPUFORT_MODULE_FILES
    global ONLY_GENERATE_KERNELS
    global ONLY_MODIFY_TRANSLATION_SOURCE
    global INCLUDE_DIRS
    global WRAP_IN_IFDEF
      
    # parse command line arguments
    parser = argparse.ArgumentParser(description="S2S translation tool for CUDA Fortran and Fortran+X")
    
    parser.add_argument("input",help="The input file.",type=str,nargs="?",default=None)
    parser.add_argument("-c,--only-create-mod-files",dest="onlyCreateGpufortModuleFiles",action="store_true",help="Only generate GPUFORT modules files. No other output is generated.")
    parser.add_argument("-s,--skip-create-mod-files",dest="skipCreateGpufortModuleFiles",action="store_true",help="Skip generating GPUFORT modules, e.g. if they already exist. Mutually exclusive")
    parser.add_argument("-o,--output", help="The output file. Interface module and HIP C++ implementation are named accordingly. GPUFORT module files are generated too.", default=sys.stdout, required=False, type=argparse.FileType("w"))
    parser.add_argument("--working-dir",dest="workingDir",default=os.getcwd(),type=str,help="Set working directory.") # shadow arg
    parser.add_argument("-d,--search-dirs", dest="searchDirs", help="Module search dir", nargs="*",  required=False, default=[], type=str)
    parser.add_argument("-w,--wrap-in-ifdef",dest="wrapInIfdef",action="store_true",help="Wrap converted lines into ifdef in host code.")
    parser.add_argument("-E,--destination-dialect",dest="destinationDialect",default=None,type=str,help="One of: {}".format(", ".join(scanner.SUPPORTED_DESTINATION_DIALECTS)))
    parser.add_argument("-m,--only-modify-host-code",dest="onlyModifyTranslationSource",action="store_true",help="Only modify host code; do not generate kernels.")
    group_hip = parser.add_argument_group('Fortran to HIP translation')
    group_hip.add_argument("-k,--only-generate-kernels",dest="onlyGenerateKernels",action="store_true",help="Only generate kernels; do not modify host code.")
    group_cuf = parser.add_argument_group('CUDA Fortran')
    group_cuf.add_argument("--cublas-v2",dest="cublasV2",action="store_true",help="Assume cublas v2 function signatures that use a handle. Overrides config value.")
    # config options: shadow arguments that are actually taken care of by raw argument parsing
    group_config = parser.add_argument_group('Config file')
    group_config.add_argument("--print-config-defaults",dest="printConfigDefaults",action="store_true",help="Print config defaults. "+\
            "Config values can be overriden by providing a config file. A number of config values can be overwritten via this CLI.")
    group_config.add_argument("--config-file",default=None,type=argparse.FileType("r"),dest="configFile",help="Provide a config file.")
    # logging
    group_logging = parser.add_argument_group('Logging')
    group_logging.add_argument("--log-level",dest="logLevel",required=False,type=str,default="",help="Set log level. Overrides config value.")
    group_logging.add_argument("-v,--verbose",dest="verbose",required=False,action="store_true",default="",help="Print all log messages to error output stream too.")
    
    parser.set_defaults(printConfigDefaults=False,dumpIndex=False,\
      wrapInIfdef=False,cublasV2=False,onlyGenerateKernels=False,onlyModifyTranslationSource=False,\
      onlyCreateGpufortModuleFiles=False,skipCreateGpufortModuleFiles=False,verbose=False)
    args, unknownArgs = parser.parse_known_args()

    if args.printConfigDefaults:
        gpufortPythonDir=os.path.dirname(os.path.realpath(__file__))
        optionsFiles = [
          "./gpufort_options.py.in",
          "scanner/scanner_options.py.in",
          "translator/translator_options.py.in",
          "fort2hip/fort2hip_options.py.in",
          "indexer/indexer_options.py.in",
          "indexer/scoper_options.py.in",
          "linemapper/linemapper_options.py.in",
          "utils/logging_options.py.in"
        ]
        print("\nCONFIGURABLE GPUFORT OPTIONS (DEFAULT VALUES):")
        for optionsFile in optionsFiles:
            prefix = optionsFile.split("/")[1].split("_")[0]
            prefix = prefix.replace("logging","utils.logging") # hack
            print("\n---- "+prefix+" -----------------------------")
            with open(gpufortPythonDir+"/"+optionsFile) as f:
                for line in f.readlines():
                   if len(line.strip()) and\
                     "# SPDX-License-Identifier:" not in line and\
                     "# Copyright (c)" not in line:
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
    ## VALIDATION
    if args.onlyCreateGpufortModuleFiles and args.skipCreateGpufortModuleFiles:
        msg = "switches '--only-create-mod-files' and '--skip-generate-mod-files' are mutually exclusive."
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
    # mutually exclusive arguments
    if args.onlyGenerateKernels and args.onlyModifyTranslationSource:
        msg = "switches '--only-generate-kernels' and 'only-modify-host-code' are mutually exclusive."
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(2)
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
    ## OVERWRITE CONFIG VALUES
    # parse file and create index in parallel
    if args.destinationDialect != None:
        scanner.DESTINATION_DIALECT = \
          scanner.checkDestinationDialect(args.destinationDialect)
    # only create modules files / skip module file generation
    if args.onlyCreateGpufortModuleFiles: 
        ONLY_CREATE_GPUFORT_MODULE_FILES = True
    if args.skipCreateGpufortModuleFiles: 
        SKIP_CREATE_GPUFORT_MODULE_FILES = True
    # only generate kernels / modify source 
    if args.onlyGenerateKernels:
        ONLY_GENERATE_KERNELS = True
    if args.onlyModifyTranslationSource:
        ONLY_MODIFY_TRANSLATION_SOURCE = True
    # wrap modified lines in ifdef
    if args.wrapInIfdef:
        linemapper.FILE_MODIFICATION_WRAP_IN_IFDEF = True
    # logging
    if len(args.logLevel):
        LOG_LEVEL = args.logLevel
    if args.verbose:
        utils.logging.VERBOSE = args.verbose
    # other
    if args.cublasV2:
        scanner.CUBLAS_VERSION = 2
        translator.CUBLAS_VERSION = 2
    return args, unknownArgs

def initLogging(inputFilepath):
    global LOG_LEVEL
    global LOG_FORMAT

    inputFilepathHash = hashlib.md5(inputFilepath.encode()).hexdigest()[0:8]
    logfileBaseName = "log-{}.log".format(inputFilepathHash)
   
    logFormat   = LOG_FORMAT.replace("%(filename)s",inputFilepath)
    logFilepath = utils.logging.initLogging(logfileBaseName,logFormat,LOG_LEVEL)
 
    msg = "input file: {0} (log id: {1})".format(inputFilepath,inputFilepathHash)
    utils.logging.logInfo(LOG_PREFIX,"initLogging",msg)
    msg = "log file:   {0} (log level: {1}) ".format(logFilepath,LOG_LEVEL)
    utils.logging.logInfo(LOG_PREFIX,"initLogging",msg)
    return logFilepath

if __name__ == "__main__":
    # read config and command line arguments
    configFilepath, includeDirs, defines = parseRawCommandLineArguments()
    if configFilepath != None:
        parseConfig(configFilepath)
    args, unknownArgs = parseCommandLineArguments()
    if len(POST_CLI_ACTIONS):
        msg = "run registered actions"
        utils.logging.logInfo(msg,verbose=False)
        for action in POST_CLI_ACTIONS:
            if callable(action):
                action(args,unknownArgs)

    # init logging
    inputFilepath = os.path.abspath(args.input)
    logFilepath   = initLogging(inputFilepath)
    
    # Update INCLUDE_DIRS from all sources    
    INCLUDE_DIRS += args.searchDirs
    INCLUDE_DIRS += includeDirs
    oneOrMoreSearchDirsNotFound = False
    for i,directory in enumerate(INCLUDE_DIRS):
        if directory[0]  != "/":
            INCLUDE_DIRS[i] = args.workingDir+"/"+directory
        if not os.path.exists(INCLUDE_DIRS[i]):
            msg = "search directory '{}' cannot be found".format(INCLUDE_DIRS[i])
            utils.logging.logError(msg,verbose=False) 
            oneOrMoreSearchDirsNotFound = True
    INCLUDE_DIRS.append(args.workingDir)
    if oneOrMoreSearchDirsNotFound:
        sys.exit(2)

    # scanner must be invoked after index creation
    if ENABLE_PROFILING:
        profiler = cProfile.Profile()
        profiler.enable()
    #
    index = createIndex(INCLUDE_DIRS,defines,inputFilepath)
    if not ONLY_CREATE_GPUFORT_MODULE_FILES:
        records = linemapper.readFile(inputFilepath,defines)
        stree   = scanner.parseFile(records,index,inputFilepath)    
 
        # extract kernels
        outputFilePrefix = ".".join(inputFilepath.split(".")[:-1])
        basename         =  os.path.basename(outputFilePrefix)
        if "hip" in scanner.DESTINATION_DIALECT: 
            kernelsToConvertToHip = ["*"]
        else:
            kernelsToConvertToHip = scanner.KERNELS_TO_CONVERT_TO_HIP
        fort2hip.createHipKernels(stree,index,kernelsToConvertToHip,outputFilePrefix,basename,\
          generateCode=not ONLY_MODIFY_TRANSLATION_SOURCE)
        
        # modify original file
        if not ONLY_GENERATE_KERNELS:
            __translateSource(inputFilepath,stree,records,index) 
    #
    if ENABLE_PROFILING:
        profiler.disable() 
        s = io.StringIO()
        sortby = 'cumulative'
        stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        stats.print_stats(PROFILING_OUTPUT_NUM_FUNCTIONS)
        print(s.getvalue())

    # shutdown logging
    msg = "log file:   {0} (log level: {1}) ".format(logFilepath,LOG_LEVEL)
    utils.logging.logInfo(LOG_PREFIX,"__main__",msg)
    utils.logging.shutdown()
