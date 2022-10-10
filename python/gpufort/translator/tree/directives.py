# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
    
# todo: move to different place
def format_directive(directive_line, max_line_width):
    result = ""
    line = ""
    tokens = directive_line.split(" ")
    sentinel = tokens[0]
    for tk in tokens:
        if len(line + tk) > max_line_width - 1:
            result += line + "&\n"
            line = sentinel + " "
        line += tk + " "
    result += line.rstrip()
    return result
