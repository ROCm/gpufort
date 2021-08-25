#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import sys
sys.path.append("../..")
from kernelconverter.grammar import * 

print (declaration.parseString("COMPLEX(DP), ALLOCATABLE :: wfcatom_d(:,:,:) ! atomic wfcs for initialization (device)"))
print (attributes.parseString("attributes(DEVICE) :: etatom_d, wfcatom_d, randy_d"))
#print (derived_type_member.parseString("threadidx%x3")[0].c_str())
#print (datatype.parseString("double precision")[0].c_str())
#print (declaration.parseString("real(kind=8) :: rhx, rhy")[0].c_str())
#print (declaration.parseString("real(kind=8), device :: rhx, rhy")[0].c_str())
#print (declaration.parseString("real, device, parameter :: rhx, rhy")[0].c_str())
#print (datatype.parseString("integer(kind=2)")[0].c_str())
#print (declaration.parseString("real, device, parameter :: rhx(:,:), rhy(:,:)")[0].c_str())
#print (declaration.parseString("real, device, parameter :: rhx(:,:) = 3, rhy(:,:) = 2")[0].c_str())