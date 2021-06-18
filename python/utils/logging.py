# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import re
import logging

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_options.py.in")).read())

__LOG_LEVEL        = "WARNING" # should only be modified by initLogging
__LOG_LEVEL_AS_INT = getattr(logging,__LOG_LEVEL)
__LOG_FORMAT       = "%(levelname)s:%(message)s"

ERR_UTILS_LOGGING_UNSUPPORTED_LOG_LEVEL  = 91001
ERR_UTILS_LOGGING_LOG_DIR_DOES_NOT_EXIST = 91002

def initLogging(logfileBaseName,logFormat,logLevel):
    """
    Init the logging infrastructure.

    :param str logFormat: The format that the log writer should use.
    :param str logfileBaseName:  The base name of the log file that this logging module should use.
    :return 
    :note: Directory for storing the log file and further options can be specified
           via global variables before calling this method.
    """
    global __LOG_LEVEL
    global __LOG_LEVEL_AS_INT
    global __LOG_FORMAT
    global LOG_DIR
    global LOG_DIR_CREATE
    __LOG_FORMAT = logFormat

    # add custom log levels:
    registerAdditionalDebugLevels()
    logDir = LOG_DIR
    if not LOG_DIR_CREATE and not os.path.exists(logDir):
        msg = "directory for storing log files ('{}') does not exist".format(logDir)
        print("ERROR: "+msg)
        sys.exit(2)
    os.makedirs(logDir,exist_ok=True)

    # TODO check if log level exists
  
    maxDebugLevel = 5 
    supportedLevels = ["ERROR","WARNING","INFO","DEBUG"] + [ "DEBUG"+str(i) for i in range(2,maxDebugLevel+1) ]
    if logLevel.upper() not in supportedLevels:
        msg = "unsupported log level: {}; must be one of (arbitrary case): {}".format(logLevel,",".join(supportedLevels))
        print("ERROR: "+msg,file=sys.stderr)
        sys.exit(ERR_UTILS_LOGGING_UNSUPPORTED_LOG_LEVEL)
    try:
        logfilePath="{0}/{1}".format(logDir,logfileBaseName)
        __LOG_LEVEL_AS_INT = getattr(logging,logLevel.upper(),getattr(logging,"WARNING"))
        __LOG_LEVEL        = logLevel
        logging.basicConfig(format=logFormat,filename=logfilePath,filemode="w", level=__LOG_LEVEL_AS_INT)
    except Exception as e:
        msg = "directoy for storing log files '{}' cannot be accessed".format(logDir)
        print("ERROR: "+msg,file=sys.stderr)
        raise e
        sys.exit(ERR_UTILS_LOGGING_LOG_DIR_DOES_NOT_EXIST)
    return logfilePath

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    ! Taken from: https://stackoverflow.com/a/35804945
    
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def registerAdditionalDebugLevels():
    addLoggingLevel("DEBUG2", logging.DEBUG-1, methodName="debug2")
    addLoggingLevel("DEBUG3", logging.DEBUG-2, methodName="debug3")
    addLoggingLevel("DEBUG4", logging.DEBUG-3, methodName="debug4")
    addLoggingLevel("DEBUG5", logging.DEBUG-4, methodName="debug5")

def __makeMessage(prefix,funcName,rawMsg):
    return prefix+"."+funcName+"(...):\t"+rawMsg

def __printMessage(levelname,message):
    print(__LOG_FORMAT.replace("%(levelname)s",levelname).\
      replace("%(message)s",message),file=sys.stderr)

def logInfo(prefix,funcName,rawMsg):
    global __LOG_LEVEL_AS_INT
    global VERBOSE
    global LOG_FILTER
    
    msg = __makeMessage(prefix,funcName,rawMsg)
    if LOG_FILTER == None or re.search(LOG_FILTER,msg):
        logging.getLogger("").info(msg)
        if VERBOSE and __LOG_LEVEL_AS_INT <= getattr(logging,"INFO"):
            __printMessage("INFO",msg)

def logError(prefix,funcName,rawMsg):
    global VERBOSE
    global LOG_FILTER
    
    msg = __makeMessage(prefix,funcName,rawMsg)
    if LOG_FILTER == None or re.search(LOG_FILTER,msg):
        logging.getLogger("").error(msg)
        __printMessage("ERROR",msg)

def logWarning(prefix,funcName,rawMsg):
    global __LOG_LEVEL_AS_INT
    global VERBOSE
    global LOG_FILTER
    
    msg = __makeMessage(prefix,funcName,rawMsg)
    if LOG_FILTER == None or re.search(LOG_FILTER,msg):
        logging.getLogger("").warning(msg)
        __printMessage("WARNING",msg)

def logDebug(prefix,funcName,rawMsg,debugLevel=1):
    global __LOG_LEVEL_AS_INT
    global VERBOSE
    global LOG_FILTER
   
    msg = __makeMessage(prefix,funcName,rawMsg)
    if LOG_FILTER == None or re.search(LOG_FILTER,msg):
        if debugLevel == 1:
           logging.getLogger("").debug(msg)
        elif debugLevel == 2:
           logging.getLogger("").debug2(msg)
        elif debugLevel == 3:
           logging.getLogger("").debug3(msg)
        elif debugLevel == 4:
           logging.getLogger("").debug4(msg)
        elif debugLevel == 5:
           logging.getLogger("").debug5(msg)
        else:
            assert False, "debug level not supported"
        if VERBOSE and __LOG_LEVEL_AS_INT <= getattr(logging,"DEBUG")-debugLevel+1:
            levelname =  "DEBUG" if ( debugLevel == 1 ) else ("DEBUG"+str(debugLevel))
            __printMessage(levelname,msg)

def logDebug1(prefix,funcName,msg):
    logDebug(prefix,funcName,msg,1)
def logDebug2(prefix,funcName,msg):
    logDebug(prefix,funcName,msg,2)
def logDebug3(prefix,funcName,msg):
    logDebug(prefix,funcName,msg,3)
def logDebug4(prefix,funcName,msg):
    logDebug(prefix,funcName,msg,4)
def logDebug5(prefix,funcName,msg):
    logDebug(prefix,funcName,msg,5)
    
def logEnterFunction(prefix,funcName,args={}):
    """
    Log entry to a function.

    :param str prefix: (sub-)package name
    :param str funcName: name of the function
    :param dict args: arguments (identifier and value) that have a meaningful string representation.
    """
    if len(args):
        addition = " [arguments: "+ ", ".join(a+"="+str(args[a]) for a in args.keys())+"]"
    else:
        addition = "" 
    logDebug(prefix,funcName,"enter"+addition)

def logLeaveFunction(prefix,funcName,returnVals={}):
    """
    Log return from a function.
    :param str prefix: (sub-)package name
    :param str funcName: name of the function
    :param dict retVals: arguments (identifier and value) that have a meaningful string representation.
    """
    if len(returnVals):
        addition = " [return values: "+ ", ".join(a+"="+str(returnVals[a]) for a in returnVals.keys())+"]"
    else:
        addition = "" 
    logDebug(prefix,funcName,"return"+addition)
