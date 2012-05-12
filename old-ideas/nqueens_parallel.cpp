#include "OpenCLSetup.hpp"
#include <string.h>


//Should make these into commandline args, lazy
#define BOARDSIZE 10 //Make sure this is <= MAX inside nqueens.cl
#define BATCHSIZE 1024
#define PRESOLVE 3
#define BUFFSIZE ((PRESOLVE+1)*BATCHSIZE)
#define LOCALSIZE 128

int unsafe(int y, int * b) {
  int x = b[y];
  int i,conflicts=0;
  for (i = 1; i <= y; i++) {
    int t = b[y - i];
    //Probably cheaper to check the whole board than to have an if in here
    conflicts += 
      (t == x ||
       t == x - i ||
       t == x + i);    
  }
  
  return conflicts;
}

int main(){
  OpenCLWrapper w;
  w.enableProfiling = true;
  cl_int b[BOARDSIZE] = {0};
  cl_int buffer[BUFFSIZE] = {0};

  int k,s=0;
  int par_solns = 0;
  int y = 0;
  int end_y = PRESOLVE, start_y = end_y+1;
  int size = BOARDSIZE;
  int batch_size = BATCHSIZE;
  int res_size = BATCHSIZE/LOCALSIZE;
  cl_int results[res_size];
  size_t globalWorkSize[1] = {BATCHSIZE};
  size_t localWorkSize[1] = {LOCALSIZE};

  int event_id=0;

  try{

    w.createContext();
    w.createCommandQueue();
    w.createProgram("nqueens.cl");
    	
    w.createKernel("solve");

    //Result array
    w.addMemObject(clCreateBuffer(w.context,CL_MEM_COPY_HOST_PTR | CL_MEM_WRITE_ONLY,
				  sizeof(cl_int)*(BATCHSIZE/LOCALSIZE),buffer,NULL));
    //Seeds
    w.addMemObject(clCreateBuffer(w.context,CL_MEM_READ_ONLY,
				  sizeof(cl_int)*BUFFSIZE,NULL,NULL));

    w.check(clSetKernelArg(w.kernels["solve"],0,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 0");
    w.check(clSetKernelArg(w.kernels["solve"],1,sizeof(cl_mem),
			   &w.memObjects[1]), "Error setting kernel arg 1");
    w.check(clSetKernelArg(w.kernels["solve"],2,sizeof(cl_int),
			   &size), "Error setting kernel arg 2");    
    w.check(clSetKernelArg(w.kernels["solve"],3,sizeof(cl_int),
			   &start_y), "Error setting kernel arg 3");    
    w.check(clSetKernelArg(w.kernels["solve"],4,sizeof(cl_int),
			   &res_size), "Error setting kernel arg 4");    


    b[y] = -1;
    while (y >= 0 ) {
      
      do {
	b[y]++;
      } 
      while ((b[y] < size) && unsafe(y,b));
      
      if (b[y] < size) {
	if (y < end_y) {
	  b[++y] = -1;
	}
	else {
	  //board is in a valid state at this point
	  memcpy(&buffer[ (par_solns)*(end_y+1) ],b,(end_y+1)*sizeof(cl_int) );
	  
	  //Buffer is full, we ship out to the device at this point
	  if(par_solns == batch_size-1){

	    //Has to be blocking, buffer is volatile
	    w.check(
		    clEnqueueWriteBuffer(w.commandQueue,w.memObjects[1],
					 CL_TRUE,0,BUFFSIZE*sizeof(cl_int),
					 buffer,0,NULL,NULL),
		    "Error enqueueing write buffer");
	    w.check(
		    clEnqueueNDRangeKernel(w.commandQueue,w.kernels["solve"],
					   1,NULL,globalWorkSize,
					   localWorkSize,0,NULL,&w.events[event_id++]),
		    "Error enqueueing kernel");
	    
	    /* Debugging
	    //Read the results
	    w.check(clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
					(BATCHSIZE/LOCALSIZE)*sizeof(cl_int),results,0,NULL,NULL),
		    "Error enqueueing read buffer");
	    for(k=0; k<BATCHSIZE/LOCALSIZE; k++){
	      printf("%i,",results[k]);
	    }
	    printf("\n");
	    */

	    par_solns = -1;
	  }
	  par_solns++;
	  
	}
      } 
      else {
	y--;
      }
    }

    if(par_solns > 0){
      //printf("%i partial solutions left over\n",par_solns);
      //Once more to flush the buffer
      w.check(
	      clEnqueueWriteBuffer(w.commandQueue,w.memObjects[1],
				   CL_TRUE,0,(par_solns*(PRESOLVE+1))*sizeof(cl_int),
				   buffer,0,NULL,NULL),
	      "Error enqueueing write buffer");
      globalWorkSize[0] = par_solns;
      localWorkSize[0] = 1;      

      w.check(
	      clEnqueueNDRangeKernel(w.commandQueue,w.kernels["solve"],
				     1,NULL,globalWorkSize,
				     localWorkSize,0,NULL,&w.events[event_id++]),
	      "Error enqueueing kernel");            
    }

    //Read the results
    w.check(clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
				(BATCHSIZE/LOCALSIZE)*sizeof(cl_int),results,0,NULL,NULL),
	    "Error enqueueing read buffer");
    

    //Sum up the results
    s=0;
    for(k=0; k<BATCHSIZE/LOCALSIZE; k++){
      //printf("%i,",results[k]);
      s+=results[k];
    }
    printf("\nFound %i solutions for %i-queens\n",s,BOARDSIZE);

    //Do some profiling and shit
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

  }
  catch ( runtime_error& e) {
    std::cerr << e.what() << std::endl;
    w.cleanup();
    return 1;    
  }  

  return 0;
}
