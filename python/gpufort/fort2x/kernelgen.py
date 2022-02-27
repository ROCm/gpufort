# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import copy

from gpufort import util
from gpufort import indexer

LOG_PREFIX = "fort2x.hip.kernelgen.KernelGeneratorBase"

class KernelGeneratorBase:
    
    def _create_kernel_base_context(self,
                                    kernel_name,
                                    c_body):
        #global GET_LAUNCH_BOUNDS
        #launch_bounds = GET_LAUNCH_BOUNDS(kernel_name)
        return {
            "name": kernel_name,
            "c_body": c_body,
            "global_vars": [],
            "global_reduced_vars": [],
            "shared_vars": [],
            "local_vars": [],
        }

    def _create_launcher_base_context(self, kernel_name, kind, problem_size,
                                      debug_code):
        return {
            "kind": kind,
            "name": "_".join(["launch", kernel_name, kind]),
            "used_modules": [],
            "problem_size": problem_size,
            "debug_code": debug_code,
        }

    def render_gpu_kernel_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_begin_kernel_comment_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_end_kernel_comment_cpp(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_begin_kernel_comment_f03(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_end_kernel_comment_f03(self):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_cpu_launcher_cpp(self, launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_launcher_interface_f03(self, launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_cpu_routine_f03(self, launcher):
        """:return: snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []

    def render_gpu_kernel_cpp(self):
        """:return: Snippets created by this routine.
        :note: to be defined by subclass.
        """
        return []