# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import scanner.scanner as scanner
import linemapper.linemapper as linemapper
import indexer.indexer as indexer
import indexer.indexerutils as indexerutils
import utils.kwargs

def parse_file(**kwargs):
    r"""Create scanner tree from file content.
    :param \*\*kwargs: See below.
    :return: a triple of scanner tree root node, (updated) index, and linemaps
    :Keyword Arguments:
        * *file_path* (``str``):
            Path to the file that should be parsed.
        * *file_content* (``str``):
            Content of the file that should be parsed.
        * *file_linemaps* (``list``):
            Linemaps of the file that should be parsed;
            see GPUFORT's linemapper component.
        * *file_is_indexed* (``bool``):
            Index already contains entries for the main file.
        * *preproc_options* (``str``): Options to pass to the C preprocessor
            of the linemapper component.
        * *other_files_paths* (``list(str)``):
            Paths to other files that contain module files that
            contain definitions required for parsing the main file.
            NOTE: It is assumed that these files have not been indexed yet.
        * *other_files_contents* (``list(str)``):
            Content that contain module files that
            contain definitions required for parsing the main file.
            NOTE: It is assumed that these files have not been indexed yet.
        * *index* (``list``):
            Index records created via GPUFORT's indexer component.
    """
    # Determine if there is an index arg and
    # what has already been indexed      
    index,have_index = utils.kwargs.get_value("index",[],**kwargs)
    file_is_indexed,_  = utils.kwargs.get_value("file_is_indexed",have_index,**kwargs)
    # handle main file
    file_content, have_file_content = utils.kwargs.get_value("file_content",None,**kwargs)
    linemaps, have_linemaps         = utils.kwargs.get_value("file_linemaps",None,**kwargs)
    file_path,have_file_path        = utils.kwargs.get_value("file_path","<unknown>",**kwargs)
    if not have_file_content and not have_linemaps:
       if not have_file_path:
           raise ValueError("Missing keyword argument: At least one of `file_path`, `file_content`, `linemaps` must be specified.")
       with open(file_path,"r") as infile:
           file_content = infile.read()
    preproc_options,_ = utils.kwargs.get_value("preproc_options","",**kwargs)
    if not have_linemaps:
       linemaps = linemapper.read_lines(file_content.split("\n"),
                                        preproc_options,
                                        file_path)
    if not file_is_indexed:
       indexer.update_index_from_linemaps(linemaps,index)
       
    # handles dependencies
    other_files_contents,have_other_files_contents = utils.kwargs.get_value("other_files_contents",[],**kwargs)
    other_files_paths,have_other_files_paths       = utils.kwargs.get_value("other_files_paths",[],**kwargs)
    if have_other_files_paths:
       for path in other_files_paths:
           with open(path,"r") as infile:
               other_files_contents.append(infile.read())
    for content in other_files_contents:
        indexerutils.update_index_from_snippet(index,
                                               content,
                                               preproc_options)
    return scanner.parse_file(linemaps,index), index, linemaps
