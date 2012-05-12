#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef queen
#define queen int
#endif

int seq_count_conflicts(__global queen *queens, int nqueens, int row, int col)
{
  int i,j,curr[2];
  int conflicts = 0;
  for(i=0, j=0; i<nqueens; i++, j+=2)
  {
    curr[0] = queens[j];
    curr[1] = queens[j+1];
    
    conflicts += ((row == curr[0]) || (col == curr[1]) || 
		  (abs_diff(row,curr[0]) == abs_diff(col,curr[1])));
    //Occupied square
    conflicts += ((row == curr[0]) && (col == curr[1]))*nqueens;
  }
  return conflicts;
}

__kernel void seq_solve(__global queen *queens,
    const int nqueens,
    const int max_iters) 
{
  int q = 0;
  int CFIs = 0; 
  int iters = 0;

  while (CFIs < nqueens && iters < max_iters)
  {
    int row = queens[q];
    int col = queens[q+1];
    int conflicts = seq_count_conflicts(queens, nqueens, row, col);
    //conflicts = conflicts - nqueens;
    
    if (conflicts != nqueens+1)
    {
      CFIs = 0;
      int r = 0;
      int min_square_r;
      //int min_square_c;
      int min_value = nqueens*nqueens;
      for(r = 0; r < nqueens; r++)
      {
        int c;
        c = seq_count_conflicts(queens, nqueens, r, col);
        if (c < min_value )
        {
          min_value = c;
          min_square_r = r;
        }
      }
      queens[q] = min_square_r;
      //queens[q+1] = col;
    }
    else { CFIs += 1; }
    q = (q+2)%nqueens;
    iters += 1;
  }
}

// vim: syntax=c
