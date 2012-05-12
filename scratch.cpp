#include <string.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#ifndef queen
#define queen int
#endif

#define BOARDSIZE 8
#define LOCALSIZE 8
#define MAX_ITERS 1000

using namespace std;

queen * random_board(); //Allocates space
void print_board(queen * q);

int main(){
  queen * queens = random_board();
  print_board(queens);
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
