#include "OpenCLSetup.hpp"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <algorithm>

#ifndef queen
#define queen cl_int
#endif

#define BOARDSIZE 100
#define LOCALSIZE 10
#define MAX_ITERS 10000

using namespace std;

queen * random_board(); //Allocates space
void print_board(queen * q);

int main(){
  srand(time(NULL));

  queen * queens = random_board(),* curr_q;
  queen conflicts[BOARDSIZE] = {0};
  queen zero[BOARDSIZE] = {0};
  int curr = 0,cf_iters = 0,iters = 0,min_con,min_c;
  int nqueens = BOARDSIZE,group_size=LOCALSIZE;
  OpenCLWrapper w;
  //w.enableProfiling = true;
  
  size_t globalWorkSize[1] = {BOARDSIZE*BOARDSIZE/LOCALSIZE};//queens,cols
  size_t localWorkSize[1] = {BOARDSIZE};

  //size_t globalWorkSize[1] = {BOARDSIZE};//queens,cols
  //size_t localWorkSize[1] = {1};
  
  try{
    
    w.createContext();
    w.createCommandQueue();
    w.createProgram("ir.cl");
    
    w.createKernel("count_conflicts");
    
    //Queen array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
				  sizeof(queen)*BOARDSIZE*2,
				  queens,NULL));
    //Result array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
				  sizeof(queen)*BOARDSIZE,
				  conflicts,NULL));
    
    w.check(clSetKernelArg(w.kernels["count_conflicts"],0,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 0");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],1,sizeof(cl_mem),
			   &w.memObjects[1]), "Error setting kernel arg 1");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],2,
			   sizeof(queen)*LOCALSIZE*2,
			   NULL), "Error setting kernel arg 2");

    w.check(clSetKernelArg(w.kernels["count_conflicts"],4,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 4");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],5,sizeof(cl_int),
			   &group_size), "Error setting kernel arg 5");
    
    do{
      curr_q = &queens[2*curr];

      w.check(clSetKernelArg(w.kernels["count_conflicts"],3,sizeof(cl_int),
			     &curr_q[0]), "Error setting kernel arg 3");
      
      w.check(
	      clEnqueueWriteBuffer(w.commandQueue,w.memObjects[0],
				   CL_FALSE,0,2*BOARDSIZE*sizeof(queen),
				   queens,0,NULL,NULL),
	      "Error enqueueing write buffer");
      w.check(
	      clEnqueueWriteBuffer(w.commandQueue,w.memObjects[1],
				   CL_FALSE,0,BOARDSIZE*sizeof(queen),
				   zero,0,NULL,NULL),
	      "Error enqueueing write buffer");

      
      w.check(
	      clEnqueueNDRangeKernel(w.commandQueue,
				     w.kernels["count_conflicts"],
				     1,0,globalWorkSize,
				     localWorkSize,0,NULL,NULL),
	      "Error enqueueing kernel");
      w.check(
	      clEnqueueReadBuffer(w.commandQueue,w.memObjects[1],CL_TRUE,0,
				  BOARDSIZE*sizeof(cl_int),conflicts,0,NULL,
				  NULL),
		    "Error enqueueing read buffer");
      
      /*
      print_board(queens);
      for(int i(0); i<BOARDSIZE; i++){
	printf("%i,",conflicts[i]);
      }
      printf("\n");
      */

      if(conflicts[curr_q[1]]-BOARDSIZE > 1){
	cf_iters = 0;
	min_con = BOARDSIZE*BOARDSIZE;
	for(int i(0); i<BOARDSIZE; i++){
	  if(conflicts[i] < min_con){
	    min_con = conflicts[i];
	    min_c = i;
	  }
	}
	queens[2*curr+1] = min_c;
      }
      
      else{cf_iters++;}

      curr = (curr+1)%BOARDSIZE;
      iters++;

    }while(cf_iters < BOARDSIZE && iters < MAX_ITERS);


   }
   catch ( runtime_error& e) {
     std::cerr << e.what() << std::endl;
     w.cleanup();
     return 1;    
   }   

  printf("Iters: %i\n",iters);

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
