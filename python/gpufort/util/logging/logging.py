# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os, sys
import re
import logging
import traceback

from . import opts

__LOG_LEVEL = "WARNING" # should only be modified by init_logging
__LOG_LEVEL_AS_INT = getattr(logging, __LOG_LEVEL)
__LOG_FORMAT = "%(levelname)s:%(message)s"
__LOG_FILE_PATH = None
__LOGGING_IS_INITIALIZED = False

ERR_UTIL_LOGGING_UNSUPPORTED_LOG_LEVEL = 91001
ERR_UTIL_LOGGING_LOG_DIR_DOES_NOT_EXIST = 91002


def shutdown():
    logging.shutdown()


def init_logging(logfile_basename="log.log",
                 log_format=__LOG_FORMAT,
                 log_level="warning"):
    """Init the logging infrastructure.

    :param str log_format: The format that the log writer should use.
    :param str logfile_basename:  The base name of the log file that this logging module should use.
    :return 
    :note: Directory for storing the log file and further options can be specified
           via global variables before calling this method.
    """
    global __LOG_LEVEL
    global __LOG_LEVEL_AS_INT
    global __LOG_FORMAT
    global __LOG_FILE_PATH
    global __LOGGING_IS_INITIALIZED
    __LOG_FORMAT = log_format

    # add custom log levels:
    _register_additional_debug_levels()
    log_dir = opts.log_dir
    if not opts.log_dir_create and not os.path.exists(log_dir):
        msg = "directory for storing log files ('{}') does not exist".format(
            log_dir)
        print("ERROR: " + msg)
        sys.exit(2)
    os.makedirs(log_dir, exist_ok=True)

    # TODO check if log level exists

    max_debug_level = 5
    supported_levels = ["ERROR", "WARNING", "INFO", "DEBUG"] + [
        "DEBUG" + str(i) for i in range(2, max_debug_level + 1)
    ]
    if log_level.upper() not in supported_levels:
        msg = "unsupported log level: {}; must be one of (arbitrary case): {}".format(
            log_level, ",".join(supported_levels))
        print("ERROR: " + msg, file=sys.stderr)
        sys.exit(ERR_UTIL_LOGGING_UNSUPPORTED_LOG_LEVEL)
    try:
        __LOG_FILE_PATH = "{0}/{1}".format(log_dir, logfile_basename)
        __LOG_LEVEL_AS_INT = getattr(logging, log_level.upper(),
                                     getattr(logging, "WARNING"))
        __LOG_LEVEL = log_level
        __LOGGING_IS_INITIALIZED = True
        logging.basicConfig(format=log_format,
                            filename=__LOG_FILE_PATH,
                            filemode="w",
                            level=__LOG_LEVEL_AS_INT)
    except Exception as e:
        msg = "directory for storing log files '{}' cannot be accessed".format(
            log_dir)
        print("ERROR: " + msg, file=sys.stderr)
        raise e
        sys.exit(ERR_UTIL_LOGGING_LOG_DIR_DOES_NOT_EXIST)
    return __LOG_FILE_PATH


def _register_additional_debug_levels(max_level=5):
    for debug_level in range(2, max_level + 1):
        label = "DEBUG" + str(debug_level)
        level = logging.DEBUG - debug_level + 1

        logging.addLevelName(level, label)
        setattr(logging, label, level)

        def log_if_level_is_active_(self, message, *args, **kwargs):
            nonlocal level
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)

        setattr(logging.getLoggerClass(), label.lower(),
                log_if_level_is_active_)


def _make_message(prefix, func_name, raw_msg):
    return prefix + "." + func_name + "(...):\t" + raw_msg


def _print_message(levelname, message):
    print(__LOG_FORMAT.replace("%(levelname)s",levelname).\
      replace("%(message)s",message),file=sys.stderr)


def log_info(prefix, func_name, raw_msg):
    global __LOGGING_IS_INITIALIZED
    global __LOG_LEVEL_AS_INT

    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    msg = _make_message(prefix, func_name, raw_msg)
    if opts.log_filter == None or re.search(opts.log_filter, msg):
        logging.getLogger("").info(msg)
        if opts.verbose and __LOG_LEVEL_AS_INT <= getattr(logging, "INFO"):
            _print_message("INFO", msg)


def log_error(prefix, func_name, raw_msg):
    global __LOGGING_IS_INITIALIZED

    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    msg = _make_message(prefix, func_name, raw_msg)
    if opts.traceback:
        stack = "".join(traceback.format_stack()[:-1])
        msg += "\n\n error site:\n\n" + stack + "\n"
    logging.getLogger("").error(msg)
    _print_message("ERROR", msg)


def log_exception(prefix, func_name, raw_msg):
    global __LOGGING_IS_INITIALIZED

    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    msg = _make_message(prefix, func_name, raw_msg)
    if opts.traceback:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        exc_stack = "".join(traceback.format_tb(exc_traceback))
        error_stack = "".join(traceback.format_stack()[:-1])
        msg += "\n\n exception site:\n\n" + exc_stack + "\n"
        msg += " error site:\n\n" + error_stack + "\n"
    logging.getLogger("").error(msg)
    _print_message("ERROR", msg)


def log_warning(prefix, func_name, raw_msg):
    global __LOGGING_IS_INITIALIZED

    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    msg = _make_message(prefix, func_name, raw_msg)
    if opts.log_filter == None or re.search(opts.log_filter, msg):
        if opts.traceback:
            stack = "".join(traceback.format_stack()[:-1])
            msg += "\n\n warning site:\n\n" + stack + "\n"
        logging.getLogger("").warning(msg)
        _print_message("WARNING", msg)


def log_debug(prefix, func_name, raw_msg, debug_level=1):
    global __LOG_LEVEL_AS_INT
    global __LOGGING_IS_INITIALIZED

    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    msg = _make_message(prefix, func_name, raw_msg)
    if opts.log_filter == None or re.search(opts.log_filter, msg):
        if debug_level == 1:
            logging.getLogger("").debug(msg)
        elif debug_level == 2:
            logging.getLogger("").debug2(msg)
        elif debug_level == 3:
            logging.getLogger("").debug3(msg)
        elif debug_level == 4:
            logging.getLogger("").debug4(msg)
        elif debug_level == 5:
            logging.getLogger("").debug5(msg)
        else:
            assert False, "debug level not supported"
        if opts.verbose and __LOG_LEVEL_AS_INT <= getattr(
                logging, "DEBUG") - debug_level + 1:
            levelname = "DEBUG" if (debug_level == 1) else ("DEBUG"
                                                            + str(debug_level))
            _print_message(levelname, msg)


def log_debug1(prefix, func_name, msg):
    log_debug(prefix, func_name, msg, 1)


def log_debug2(prefix, func_name, msg):
    log_debug(prefix, func_name, msg, 2)


def log_debug3(prefix, func_name, msg):
    log_debug(prefix, func_name, msg, 3)


def log_debug4(prefix, func_name, msg):
    log_debug(prefix, func_name, msg, 4)


def log_debug5(prefix, func_name, msg):
    log_debug(prefix, func_name, msg, 5)


def log_enter_function(prefix, func_name, args={}):
    """
    Log entry to a function.

    :param str prefix: (sub-)package name
    :param str func_name: name of the function
    :param dict args: arguments (identifier and value) that have a meaningful string representation.
    """
    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    if len(args):
        addition = " [arguments: " + ", ".join(
            a + "=" + str(args[a]) for a in args.keys()) + "]"
    else:
        addition = ""
    log_debug(prefix, func_name, "enter" + addition)


def log_leave_function(prefix, func_name, retvals={}):
    """
    Log return from a function.
    :param str prefix: (sub-)package name
    :param str func_name: name of the function
    :param dict retvals: arguments (identifier and value) that have a meaningful string representation.
    """
    if not __LOGGING_IS_INITIALIZED:
        init_logging()

    if len(retvals):
        addition = " [return values: " + ", ".join(
            a + "=" + str(retvals[a]) for a in retvals.keys()) + "]"
    else:
        addition = ""
    log_debug(prefix, func_name, "return" + addition)


def log_entry_and_exit(prefix,
                       print_args=False,
                       print_kwargs=False,
                       print_retvals=False):
    """Decorator for logging entry into function and exit from it.

    :Example Usage:

    ```python
    @gpufort.util.logging.log_entry_and_exit("myprefix")
    def myfun(a,b,c):
        # ...
    ```
    """

    def inner1(func):

        def inner2(*args, **kwargs):
            enter_args = {}
            if print_args and len(*args):
                enter_args += [{
                    "#{}".format(i): a
                } for i, a in enumerate(args)]
            if print_kwargs and len(**kwargs):
                enter_args.update(kwargs)
            log_enter_function(prefix, func.__name__, enter_args)
            #
            retvals = func(*args, **kwargs)
            #
            leave_args = {}
            if print_retvals and len(retvals):
                leave_args += [{
                    "#{}".format(i): a
                } for i, a in enumerate(args)]
            log_leave_function(prefix, func.__name__, leave_args)
            return retvals

        return inner2

    return inner1