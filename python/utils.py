# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import subprocess
import logging

#CLANG_FORMAT_STYLE="\"{BasedOnStyle: llvm, ColumnLimit: 140}\""
CLANG_FORMAT_STYLE="\"{BasedOnStyle: llvm, ColumnLimit: 140, BinPackArguments: false, BinPackParameters: false, AllowAllArgumentsOnNextLine: false, AllowAllParametersOfDeclarationOnNextLine: false}\""

def prettifyCCode(cCode):
    """
    Requires clang-format 7.0+.
    
    Use e.g. the one coming with ROCm:
    /opt/rocm-<suffix>/hcc/bin/clang-format"
    """
    command = "printf \"{0}\" | clang-format -style={1}".format(cCode.replace("%","%%"),CLANG_FORMAT_STYLE)
    return subprocess.check_output(command,shell=True).decode('ascii')

def prettifyCFile(cPath):
    """
    Requires clang-format 7.0+
    
    Use e.g. the one coming with ROCm:
    /opt/rocm-<suffix>/hcc/bin/clang-format"
    """
    command = "clang-format -i -style={1} {0}".format(cPath,CLANG_FORMAT_STYLE) # writes inplace
    subprocess.check_output(command,shell=True).decode('ascii')

def prettifyFCode(fCode):
    """
    Requires fprettify
    """
    command = "printf \"{0}\"".format(fCode.replace("%","%%"))
    command += r"| fprettify -l 1000 | sed 's,\s*\([<>]\)\s*[<>]\s*[<>]\s*,\1\1\1,g'"
    return subprocess.check_output(command,shell=True).decode('ascii')

def prettifyFFile(fPath):
    """
    Requires fprettify
    """
    subprocess.check_call(["fprettify","-l 1000",fPath], stdout=subprocess.DEVNULL)
    subprocess.check_call(r"sed -i 's,\s*\([<>]\)\s*[<>]\s*[<>]\s*,\1\1\1,g' {0}".format(fPath), shell=True, stdout=subprocess.DEVNULL)
    #return subprocess.check_output(command,shell=True).decode('ascii')
    #pass

def readCFileWithoutComments(filepath,unifdefArgs=""):
    """
    Requires gcc, unifdef
    """
    command="gcc -w -fmax-errors=100 -fpreprocessed -dD -E {0}".format(filepath)
    if len(unifdefArgs):
        command += " | unifdef {0}".format(unifdefArgs)
    try: 
       output = subprocess.check_output(command,shell=True).decode('UTF-8')
    except subprocess.CalledProcessError as cpe:
       if not cpe.returncode == 1: # see https://linux.die.net/man/1/unifdef
           raise cpe
       else:
           output = cpe.output.decode("UTF-8")
    return output

def readCFile(filepath,unifdefArgs=""):
    """
    Requires unifdef
    """
    command="cat {0}".format(filepath)
    if len(unifdefArgs):
        command += " | unifdef {0}".format(unifdefArgs)
    try: 
       output = subprocess.check_output(command,shell=True).decode('UTF-8')
    except subprocess.CalledProcessError as cpe:
       if not cpe.returncode == 1: # see https://linux.die.net/man/1/unifdef
           raise cpe
       else:
           output = cpe.output.decode("UTF-8")
    return output

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
