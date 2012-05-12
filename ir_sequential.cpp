#include "OpenCLSetup.hpp"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <algorithm>

#ifndef queen
#define queen cl_int
#endif

#define BOARDSIZE 8
#define MAX_ITERS 10000

using namespace std;

queen * random_board(); //Allocates space
void print_board(queen * q);

int main(){
  srand(15);

  queen * queens = random_board(),* curr_q;
  queen conflicts[BOARDSIZE] = {0};
  queen zero[BOARDSIZE] = {0};
  int curr = 0,cf_iters = 0,iters = 0,min_con,min_c;
  int nqueens = BOARDSIZE;
  OpenCLWrapper w;
  //w.enableProfiling = true;
  
  print_board(queens);
  printf("\n");
  fflush(stdout);

  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};
 
  try{
    
    w.createContext();
    w.createCommandQueue();
    w.createProgram("sequential.cl");
    
    w.createKernel("seq_solve");
    
    //Queen array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				  sizeof(queen)*BOARDSIZE*2,
				  queens,NULL));
    
    w.check(clSetKernelArg(w.kernels["seq_solve"],0,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 0");

    w.check(clSetKernelArg(w.kernels["seq_solve"],1,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 1");
    int max = MAX_ITERS;
    printf("Max iterations: %d\n", max);
    fflush(stdout);
    w.check(clSetKernelArg(w.kernels["seq_solve"],2,sizeof(cl_int),
          &max), "Error setting kernel arg 2");

    w.check(clEnqueueWriteBuffer(w.commandQueue,w.memObjects[0],
          CL_FALSE,0,2*BOARDSIZE*sizeof(queen),
          queens,0,NULL,NULL),
          "Error enqueueing write buffer");
    
    w.check(clEnqueueNDRangeKernel(w.commandQueue,
          w.kernels["seq_solve"],
          1,0,globalWorkSize,
          localWorkSize,0,NULL,NULL),
          "Error enqueueing kernel");
    
    w.check(clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
          BOARDSIZE*2*sizeof(cl_int),queens,0,NULL,
          NULL),
          "Error enqueueing read buffer");
    
    printf("TESTING\n");
    print_board(queens);
    fflush(stdout);

    /*
       print_board(queens);
       for(int i(0); i<BOARDSIZE; i++){
       printf("%i,",conflicts[i]);
       }
       printf("\n");
       */

  }
  catch ( runtime_error& e) {
    std::cerr << e.what() << std::endl;
    w.cleanup();
    return 1;    
  }   

  //printf("Iters: %i\n",iters);

}

queen * random_board(){
  vector<queen> b(BOARDSIZE);
  for(int i(0); i<BOARDSIZE; i++){b[i] = i;}
  random_shuffle(b.begin(),b.end());

  queen * queens = new queen[2*BOARDSIZE];
  for(int q(0), j(0); q<BOARDSIZE; q++, j+=2){
    queens[j] = q;
    queens[j+1] = b[q];
  }

  return queens;
}


void print_board(queen * queens){
  char tmp[BOARDSIZE][BOARDSIZE+1];
  memset(tmp,'.',BOARDSIZE*(BOARDSIZE+1));

  for(int i(0); i<BOARDSIZE; i++){
    tmp[queens[2*i]][queens[2*i+1]] = 'Q';
    tmp[i][BOARDSIZE] = '\0';
  }

  for(int i(0); i<BOARDSIZE; i++){
    printf("%s\n",&tmp[i]);
  } 
}
