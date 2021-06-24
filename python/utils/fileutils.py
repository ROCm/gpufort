# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import subprocess
import logging
import sys

#CLANG_FORMAT_STYLE="\"{BasedOnStyle: llvm, ColumnLimit: 140}\""

def prettifyCCode(cCode,style):
    """
    Requires clang-format 7.0+.
    
    Use e.g. the one coming with ROCm:
    /opt/rocm-<suffix>/hcc/bin/clang-format"
    """
    command = "printf \"{0}\" | clang-format -style={1}".format(cCode.replace("%","%%"),style)
    return subprocess.check_output(command,shell=True).decode('ascii')

def prettifyCFile(cPath,style):
    """
    Requires clang-format 7.0+
    
    Use e.g. the one coming with ROCm:
    /opt/rocm-<suffix>/hcc/bin/clang-format"
    """
    command = "clang-format -i -style={1} {0}".format(cPath,style) # writes inplace
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
