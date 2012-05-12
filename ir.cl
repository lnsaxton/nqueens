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

  /*
  event_t e; //Copy relevant queens into local space
  e = async_work_group_copy(ws,&queens[group_size*q*2],2*group_size,0);
  wait_group_events(1,&e);
  */

  int i,j,conflicts = 0,curr[2];
  for(i = 0, j = 0; i<group_size; i++, j+=2){
    curr[0] = ws[j];
    curr[1] = ws[j+1];

    //curr[0] = queens[q*group_size*2+j];
    //curr[1] = queens[q*group_size*2+j+1];

    //OpenCL spec says that the result of boolean expressions is 0 or 1
    conflicts += ((row == curr[0]) || (col == curr[1]) || 
		  (abs_diff(row,curr[0]) == abs_diff(col,curr[1])));
    //Occupied square
    conflicts += ((row == curr[0]) && (col == curr[1]))*nqueens;
  }
  
  if(conflicts > 0){
    atom_add(&result[col],conflicts);
  }

}

