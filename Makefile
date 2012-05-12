# note, this Makefile works for cuda.acad.cis.udel.edu
#  no guarantees that it works for any other opencl system
CXX = g++
CXX_FLAGS = -I/software/cuda-sdk/OpenCL/common/inc -lrt -lOpenCL
SRUNX = /software/slurm/bin/srun -N1 --gres=gpu:1
CC = gcc


all: ir_parallel

ir_parallel: ir_parallel.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)

run:
	srun -N1 --gres=gpu:1 ./ir_parallel

clean:
	rm -f $(ALL) *~
