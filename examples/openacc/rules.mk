FC = gfortran
FOPTS ?= -std=f2008 -ffree-line-length-none -D__GPUFORT

GPUFORT_ACC_DIR = /home/amd/dominic/gpufort/tools/acc_runtime
ACC_INC = -I/$(GPUFORT_ACC_DIR)/include
ACC_LIB = -L/$(GPUFORT_ACC_DIR)/lib -lgpufort_acc        

