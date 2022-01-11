# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
def get_linemaps_content(linemaps,
                         key,
                         first_linemap_first_elem=0,
                         last_linemap_last_elem=-1):
    """Collects entries for the given key from the node's linemaps."""
    result = []
    for i,linemap in enumerate(linemaps):
        if len(linemaps)==1:
            if last_linemap_last_elem == -1:
                result += linemap[key][first_linemap_first_elem:]
            else:
                result += linemap[key][first_linemap_first_elem:last_linemap_last_elem+1]
        elif i == 0: 
            result += linemap[key][first_linemap_first_elem:]
        elif i == len(linemaps)-1:
            if last_linemap_last_elem == -1:
                result += linemap[key]
            else:
                result += linemap[key][0:last_linemap_last_elem+1]
        else:
            result += linemap[key]
    return result

def get_statement_bodies(linemaps,
                         first_linemap_first_elem=0,
                         last_linemap_last_elem=-1):
    return [stmt["body"] for stmt in get_linemaps_content(linemaps,"statements",first_linemap_first_elem,last_linemap_last_elem)] 
