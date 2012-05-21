#include "OpenCLSetup.hpp"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <algorithm>

#ifndef queen
#define queen cl_int
#endif

/*
#define BOARDSIZE 8
#define LOCALSIZE 1
#define MAX_ITERS 200
#define SEED 1336866913 //time(NULL)
#define CHECK_ITERS 1

#define REDUCE_LOCAL 2
#define REDUCE_LEN BOARDSIZE/REDUCE_LOCAL
*/

#define BOARDSIZE 128
#define LOCALSIZE 32
#define MAX_ITERS 1000
#define SEED 1336868812 //time(NULL)
#define CHECK_ITERS 1

#define REDUCE_LOCAL 2
#define REDUCE_LEN BOARDSIZE/REDUCE_LOCAL



using namespace std;

queen * random_board(); //Allocates space
void print_board(queen * q);

int main(){
  int seed = SEED;
  srand(seed);
  printf("Seed: %u\n",seed);

  queen * queens = random_board(),* curr_q;
  queen conflicts[REDUCE_LEN],indexes[REDUCE_LEN],all_conflicts[BOARDSIZE];
  queen zero[BOARDSIZE] = {0};
  int curr = 0,cf_iters = 0,iters = 0,min_con,min_c;
  int nqueens = BOARDSIZE,group_size=LOCALSIZE,reduce_len=REDUCE_LEN;
  int pseudo_rand;
  OpenCLWrapper w;
  w.enableProfiling = true;
  
  int event_id = 0;
  
  size_t globalWorkSize[1] = {BOARDSIZE*BOARDSIZE/LOCALSIZE};
  size_t localWorkSize[1] = {BOARDSIZE};  

  size_t reduce_globalWorkSize[1] = {BOARDSIZE};//queens,cols
  size_t reduce_localWorkSize[1] = {REDUCE_LOCAL};

  size_t worksize_one[1] = {1};
  
  bool done = false;
  try{
    
    w.createContext();
    w.createCommandQueue();
    w.createProgram("ir.cl");
    
    w.createKernel("count_conflicts");
    w.createKernel("reduce");
    w.createKernel("make_move");
    
    //Queen array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				  sizeof(queen)*BOARDSIZE*2,
				  queens,NULL));
    //Result array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				  sizeof(queen)*BOARDSIZE,
				  zero,NULL));
    
    //Extra (result) arrays for reduce
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_READ_WRITE,
				  sizeof(queen)*REDUCE_LEN,
				  NULL,NULL));
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				  sizeof(queen)*REDUCE_LEN,
				  zero,NULL));

    //cf_iter counter
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_READ_WRITE,
				  sizeof(queen),
				  NULL,NULL));
    
    
    w.check(clSetKernelArg(w.kernels["count_conflicts"],0,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 0");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],1,sizeof(cl_mem),
			   &w.memObjects[1]), "Error setting kernel arg 1");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],2,
			   sizeof(queen)*LOCALSIZE*2,
			   NULL), "Error setting kernel arg 2");
    /*w.check(clSetKernelArg(w.kernels["count_conflicts"],3,sizeof(cl_int),
      &nqueens), "Error setting kernel arg 3");*/
    w.check(clSetKernelArg(w.kernels["count_conflicts"],4,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 4");
    w.check(clSetKernelArg(w.kernels["count_conflicts"],5,sizeof(cl_int),
			   &group_size), "Error setting kernel arg 5");


    //Reduce kernel args
    w.check(clSetKernelArg(w.kernels["reduce"],0,sizeof(cl_mem),
			   &w.memObjects[1]), "Error setting kernel arg 0");
    w.check(clSetKernelArg(w.kernels["reduce"],1,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 1");
    /*w.check(clSetKernelArg(w.kernels["reduce"],2,sizeof(cl_int),
      &nqueens), "Error setting kernel arg 2");*/
    w.check(clSetKernelArg(w.kernels["reduce"],3,
			   sizeof(queen)*REDUCE_LOCAL,
			   NULL), "Error setting kernel arg 3");
    w.check(clSetKernelArg(w.kernels["reduce"],4,
			   sizeof(queen)*REDUCE_LOCAL,
			   NULL), "Error setting kernel arg 4");
    w.check(clSetKernelArg(w.kernels["reduce"],5,sizeof(cl_mem),
			   &w.memObjects[2]), "Error setting kernel arg 5");
    w.check(clSetKernelArg(w.kernels["reduce"],6,sizeof(cl_mem),
			   &w.memObjects[3]), "Error setting kernel arg 6");
    w.check(clSetKernelArg(w.kernels["reduce"],7,sizeof(cl_mem),
			   &w.memObjects[4]), "Error setting kernel arg 7");

    //make_move args
    w.check(clSetKernelArg(w.kernels["make_move"],0,sizeof(cl_mem),
			   &w.memObjects[2]), "Error setting kernel arg 0");
    w.check(clSetKernelArg(w.kernels["make_move"],1,sizeof(cl_mem),
			   &w.memObjects[3]), "Error setting kernel arg 1");
    w.check(clSetKernelArg(w.kernels["make_move"],2,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 2");
    /*w.check(clSetKernelArg(w.kernels["make_move"],3,sizeof(cl_mem),
			   &w.memObjects[4]), "Error setting kernel arg 3");*/
    w.check(clSetKernelArg(w.kernels["make_move"],4,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 5");   
    w.check(clSetKernelArg(w.kernels["make_move"],5,sizeof(cl_int),
			   &reduce_len), "Error setting kernel arg 6");    

    /*
    printf("Board: %i\n",0);
    print_board(queens);
    */

    do{

      curr_q = &queens[2*curr];

      w.check(clSetKernelArg(w.kernels["count_conflicts"],3,sizeof(cl_int),
			     &curr_q[0]), "Error setting kernel arg 3");
      w.check(clSetKernelArg(w.kernels["reduce"],2,sizeof(cl_int),
			     &curr_q[0]), "Error setting kernel arg 2");
      w.check(clSetKernelArg(w.kernels["reduce"],8,sizeof(cl_int),
			     &curr_q[1]), "Error setting kernel arg 8");
      pseudo_rand = rand()%2;
      w.check(clSetKernelArg(w.kernels["reduce"],9,sizeof(cl_int),
			     &pseudo_rand), "Error setting kernel arg 9");

      w.check(clSetKernelArg(w.kernels["make_move"],3,sizeof(cl_int),
			     &curr), "Error setting kernel arg 4");
      pseudo_rand = rand()%2;
      w.check(clSetKernelArg(w.kernels["make_move"],6,sizeof(cl_int),
			     &pseudo_rand), "Error setting kernel arg 4");


      //printf("row: %i\n",curr_q[0]);
      //Zero out conflicts count

      w.check(
	      clEnqueueWriteBuffer(w.commandQueue,w.memObjects[1],
				   CL_FALSE,0,BOARDSIZE*sizeof(queen),
				   zero,0,NULL,NULL),
	      "Error enqueueing write buffer");

      //Count conflicts
      w.check(
	      clEnqueueNDRangeKernel(w.commandQueue,
				     w.kernels["count_conflicts"],
				     1,0,globalWorkSize,
				     localWorkSize,0,NULL,NULL),
	      "Error enqueueing count kernel");
      //Reduce
      w.check(
	      clEnqueueNDRangeKernel(w.commandQueue,
				     w.kernels["reduce"],
				     1,0,reduce_globalWorkSize,
				     reduce_localWorkSize,0,NULL,NULL),
	      "Error enqueueing reduce kernel");

      //Make move
      w.check(
	      clEnqueueNDRangeKernel(w.commandQueue,
				     w.kernels["make_move"],
				     1,0,worksize_one,
				     worksize_one,0,NULL,NULL),
	      "Error enqueueing make move kernel");

      //Occasionally check to see if we're done
      if(iters%CHECK_ITERS == 0){
	w.check(
		clEnqueueReadBuffer(w.commandQueue,w.memObjects[4],CL_TRUE,0,
				    sizeof(cl_int),&cf_iters,0,NULL,
				    NULL),
		"Error enqueueing read buffer");
	//printf("cf_iters: %i\n",cf_iters);
	if(cf_iters >= BOARDSIZE){
	  w.check(
		  clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
				  BOARDSIZE*2*sizeof(cl_int),queens,0,NULL,
				  NULL),
		    "Error enqueueing read buffer");
	  done = true;
	}
      }

      /*
      w.check(
	      clEnqueueReadBuffer(w.commandQueue,w.memObjects[3],CL_TRUE,0,
				  REDUCE_LEN*sizeof(cl_int),indexes,0,NULL,
				  NULL),
		    "Error enqueueing read buffer");

      w.check(
	      clEnqueueReadBuffer(w.commandQueue,w.memObjects[2],CL_TRUE,0,
				  REDUCE_LEN*sizeof(cl_int),conflicts,0,NULL,
				  NULL),
		    "Error enqueueing read buffer");

      w.check(
	      clEnqueueReadBuffer(w.commandQueue,w.memObjects[1],CL_TRUE,0,
				  BOARDSIZE*sizeof(cl_int),all_conflicts,0,
				  NULL,NULL),
		    "Error enqueueing read buffer");
      */
      w.check(
	      clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
				  BOARDSIZE*2*sizeof(cl_int),queens,0,
				  NULL,NULL),
	      "Error enqueueing read buffer");      

      /*
      for(int i(0); i<BOARDSIZE; i++){
	printf("%i,",all_conflicts[i]);
      }
      printf("\n");
      for(int i(0); i<REDUCE_LEN; i++){
	printf("Square (%i,%i) has %i conflicts\n",
	       curr_q[0],indexes[i],conflicts[i]);
      }
      printf("\n");
      printf("Board: %i\n",iters+1);
      print_board(queens);
      */

      /*
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
      */

      curr = (curr+1)%BOARDSIZE;
      iters++;

    }while(!done && iters < MAX_ITERS);


  }
  catch ( runtime_error& e) {
     std::cerr << e.what() << std::endl;
     w.cleanup();
     return 1;    
   }   
  //Profiling
    cl_ulong start,end,diff;
    unsigned long max=0,min=-1,total=0;
    float avg;
    printf("%i events\n",event_id);
    for(k = 0; k<event_id; k++){
      w.check(clGetEventProfilingInfo(w.events[k],CL_PROFILING_COMMAND_START,
				      sizeof(cl_ulong),&start,NULL),
	      "Error getting profiling info");
      w.check(clGetEventProfilingInfo(w.events[k],CL_PROFILING_COMMAND_END,
				      sizeof(cl_ulong),&end,NULL),
	      "Error getting profiling info");
      diff = end-start;
      total += diff;
      avg += (float)(diff)/(float)(event_id);
      if(diff > max){
	max = diff;
      }
      if(diff < min){
	min = diff;
      }
    }
    
    printf("Total GPU computation time: %lu ns\n",total);
    printf("Average kernel execution time: %f ns\n",avg);
    printf("Max kernel execution time: %lu ns\n",max);
    printf("Min kernel execution time: %lu ns\n",min);
    
  // catch ( runtime_error& e) {
    // std::cerr << e.what() << std::endl;
     //w.cleanup();
     //return 1;    
   //}   

  if(done){
    printf("Solved board of size %i in %i iterations.\n",BOARDSIZE,iters);
    print_board(queens);
  }


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
