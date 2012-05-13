// -*- c++ -*-
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef queen
#define queen int
#endif

/*
  Each workgroup has a subset of the queens and 1 processor for each square
  Each workitem counts the number of conflicts for each queen in the group
    and then does an atomic add into the result vector

  To avoid a jump in the internal loop, if an occupied square is checked then
    nqueens is added to its conflicts: this marks the square as occupied and
    will prevent it from being chosen as the min square

*/
__kernel void count_conflicts(const __global queen * queens,
			      __global queen * result,
			      __local queen * ws,
			      const int row, const int nqueens, 
			      const int group_size ){
  int q = get_group_id(0);
  int col = get_local_id(0);

  
  event_t e; //Copy relevant queens into local space
  e = async_work_group_copy(ws,&queens[group_size*q*2],2*group_size,0);
  wait_group_events(1,&e);
  

  int i,j,conflicts = 0,curr[2];
  for(i = 0, j = 0; i<group_size; i++, j+=2){
    curr[0] = ws[j];
    curr[1] = ws[j+1];

    //curr[0] = queens[q*group_size*2+j];
    //curr[1] = queens[q*group_size*2+j+1];

    conflicts += ((row == curr[0]) || (col == curr[1]) || 
		  (abs_diff(row,curr[0]) == abs_diff(col,curr[1])));
    conflicts += ((row == curr[0]) && (col == curr[1]))*nqueens;
  }
  
  if(conflicts > 0){
    atom_add(&result[col],conflicts);
  }

}


//Number of work-items should be a power of 2
//Inspired by/borrowed from http://developer.amd.com/documentation/articles/Pages/OpenCL-Optimization-Case-Study-Simple-Reductions_3.aspx
__kernel void reduce(__global queen * conflicts,
		     const int nqueens,const int which,
		     __local queen * scratch,
		     __local queen * indexes,
		     __global queen * result,
		     __global int * result_i,
		     __global int * cf_iters,
		     const int col,const int rand){

  int gid = get_global_id(0);
  if(gid == 0){
    if( (conflicts[col]) == nqueens+1){
      cf_iters[0] = (cf_iters[0]+1);
      conflicts[col] = -1;
    }
    else{
      cf_iters[0] = 0;
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  int lid = get_local_id(0);
  int ind = gid < nqueens ? gid : -1;
  if(gid < nqueens){
      scratch[lid] = conflicts[gid];
  }
  else{
    scratch[lid] = nqueens*nqueens;
  }
  indexes[lid] = ind;
  barrier(CLK_LOCAL_MEM_FENCE);

  int offset, local_size = get_local_size(0)/2;
  for(offset = 1; offset > 0; offset >>=1 ){
    if(lid < offset){
      int other = lid + offset;
      int mine = lid;
      if(scratch[mine] < scratch[other] || 
	 (rand && scratch[mine] == scratch[other])){
	indexes[mine] = ind;
      }
      else{
	scratch[mine] = scratch[other];
	indexes[mine] = indexes[other];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if(lid == 0){
    int groupid = get_group_id(0);
    result[groupid] = scratch[0];
    result_i[groupid] = indexes[0];
  }
}

/*
  Only 1 of these should ever be spawned at once. Changing 1 value in global
  mem is probably faster than IO with host (read,change,write)
 
  Looks at the reduced list of conflicts, selects the lowest 
*/
__kernel void make_move(const __global queen * conflicts,
			const __global queen * indexes,
			__global queen * queens,
			const int which, const int nqueens,
			const int size, const int rand){

  int i,min_c=nqueens*nqueens,min_i;
  for(i = 0; i < size && min_c > 1; i++){
    if(conflicts[i] < min_c ||
       (conflicts[i] == min_c && rand)){
      min_c = conflicts[i];
      min_i = indexes[i];
    }
  }

  if(min_c != -1){
    queens[2*which+1] = min_i;
  }

}
