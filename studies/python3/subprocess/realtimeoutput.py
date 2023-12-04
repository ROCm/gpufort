#!/usr/bin/env python3
import os,sys
import subprocess

def run_subprocess(cmd):
    p = subprocess.Popen(cmd,
                         shell = False,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.STDOUT)
    output = []
    for line in iter(p.stdout.readline, b''):
        output.append(line.decode("utf-8"))
    return output

print(run_subprocess(["ls","-l"]))
print(run_subprocess(["ls","fail"]))
