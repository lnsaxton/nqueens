#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int s = 0;

static inline int unsafe(int y, int * b) {
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

/*
  This counts all solutions starting with a partially solved
  board b. The board given should be valid from columns 0 thru start_y-1
  Uses backtracking, nothing fancy
  This and unsafe(...) will be our kernel code
 */
void solve(int size,int start_y, int * b){
  int y = start_y;

  b[y] = -1;
  while (y >= start_y) {
    
    do {
      b[y]++;
    } 
    while ((b[y] < size) && unsafe(y,b));
    
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
  
}

void print_buffered_solutions(int * b,int bsize,int batch_size){
  int i,j,k=0;
  for(i=0; i<batch_size; i++){
    for(j=0; j<bsize; j++){
      printf("%i,",b[k]);
      k++;
    }
    printf("\n");
  }
}

/*
  Generates partially solved boards: from 0 to end_y is valid
  for each board. Stores parsolved boards in a buffer
  This will be our host code
*/
void batch_solve(int size,int end_y, int batch_size,int * b){
  int k;
  int scratch[size];
  int par_solns = 0;
  int y = 0;
  int buffer[(end_y+1)*batch_size];
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
	memcpy(&buffer[ (par_solns)*(end_y+1) ],b,(end_y+1)*sizeof(int) );

	//Buffer is full, we ship out to the device at this point
	if(par_solns == batch_size-1){
	  //print_buffered_solutions(buffer,end_y+1,batch_size);	  


	  //This part is just here to demonstrate correctness
	  for(k=0; k<batch_size; k++){
	    memset(scratch,0,sizeof(int)*size);
	    memcpy(scratch,&buffer[k*(end_y+1)],(end_y+1)*sizeof(int) );
	    solve(size,end_y+1,scratch);
	  }

	  par_solns = -1;
	}
	par_solns++;

      }
    } 
    else {
      y--;
    }
  }

  //Once more to flush the buffer
  for(k=0; k<par_solns; k++){
    memset(scratch,0,sizeof(int)*size);
    memcpy(scratch,&buffer[k*(end_y+1)],(end_y+1)*sizeof(int) );
    solve(size,end_y+1,scratch);
  }
  
}


void main() {
  int i,j;
  int board[8] = {-1,0,0,0,0,0,0,0};
  
  //solve(8,0,board);

 
  batch_solve(9,3,50,board);
  printf("%i solutions\n",s);

  /* Demonstration of decomposition: board is solved up to y = 1

  for(i=0; i<8; i++){
    board[0] = i;
    solve(8,1,board);
    printf("%i solutions\n",s);
  }
  */

}
