# note, this Makefile works for cuda.acad.cis.udel.edu
#  no guarantees that it works for any other opencl system
CXX = g++
CXX_FLAGS = -I/software/cuda-sdk/OpenCL/common/inc -lrt -lOpenCL
SRUNX = /software/slurm/bin/srun -N1 --gres=gpu:1
CC = gcc

ALL = nqueens_seq nqueens_parallel ir_parallel

all: $(ALL)

nqueens_seq: nqueens_seq.c
	$(CC) -o $@ $^ 

nqueens_parallel: nqueens_parallel.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)

ir_parallel: ir_parallel.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)
clean:
	rm -f $(ALL) *~