
import subprocess

from . import logging

def run_subprocess(cmd,verbose=False):
    """Run the subprocess in a blocking manner, collect error code,
    standard output and error output. 
    """
    logging.log_info("util.subprocess", "run_subprocess", " ".join(cmd))
    if verbose:
        print(cmd)
     
    p = subprocess.Popen(cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    status = p.wait()
    return status, p.stdout.read().decode("utf-8"), p.stderr.read().decode(
        "utf-8")
