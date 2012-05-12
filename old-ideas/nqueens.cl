// -*- c++ -*-
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX 16 //max problem size, me being lazy

int unsafe(int y, int * b) {
  int x = b[y];
  int i,c=0;
  for (i = 1; i <= y; i++) {
    int t = b[y - i];
    c+= (t == x ||
	 t == x - i ||
	 t == x + i);
  }
  
  return c;
}

__kernel void solve(__global int * result, const __global int * seeds,
		    const int size, const int start_y, const int res_size  ){

  int gid = get_global_id(0); //Points to a particular seed

  int b[MAX],s=0;

  //Copy our seed into private memory
  int i;
  for(i=0; i<start_y; i++){
    b[i] = seeds[gid*(start_y)+i];
  }

  int y = start_y;
  int usafe = 0;
  b[y] = -1;
  while (y >= start_y) {
    
    do {
      b[y]++;
      usafe = unsafe(y,b);
    } 
    while ((b[y] < size) && usafe);
    
    if (b[y] < size) {
      if (y < (size-1)) {
	b[++y] = -1;
      }
      else {
	//board is in solved state at this point
	s++;
      }
    } 
    else {
      y--;
    }
  }
  
  if(s>0){
    atom_add(&result[gid%res_size],s);
  }
}
