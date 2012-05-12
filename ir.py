from random import shuffle
from string import join

boardsize = 10
iter_max = 1000 #not guaranteed to find a solution, cap iterations
"""
a board is a list of n tuples representing the (r,c) position of each queen
"""

def random_board():
    board = range(boardsize)
    shuffle(board)
    queens = [ (board[i],i) for i in range(boardsize) ]
    return queens

def count_conflicts(queens,row,col):
    conflicts = 0
    occupied = False
    for q in queens:
        if (row,col) != q:
            conflicts += int((row == q[0]) or (col == q[1]) \
                                 or (abs(row-q[0]) == abs(col-q[1])))
        else:
            occupied = True

    return (conflicts,occupied)

def count_conflicts2(queens,row,col):
    conflicts = 0
    occupied = False
    for q in queens:
        #if (row,col) != q:
        conflicts += int((row == q[0]) or (col == q[1]) \
                             or (abs(row-q[0]) == abs(col-q[1])))
        conflicts += boardsize*int( (row,col) == q )
        #else:
        #    occupied = True

    return (conflicts-1,occupied)


def print_board(queens):
    s = [["."]*boardsize for i in range(boardsize)]
    for q in queens:
        s[q[0]][q[1]] = "Q"
    for r in s: print "".join(r)

def solve(queens):
    q = 0
    #If we go boardsize consecutive iterations with no conflicts, we're done
    conflict_free_iters = 0
    iters = 0

    while conflict_free_iters < boardsize and iters<iter_max:
        queen = queens[q]
        (conflicts,occupied) = count_conflicts2(queens,queen[0],queen[1])
        conflicts -= boardsize
        #Find the square in this column with the smallest number of conflicts
        if conflicts > 0:
            conflict_free_iters = 0
            r = 0
            min_square = None
            min_value = boardsize**2 #can't have this many conflicts
            while r < boardsize:
                (c,occupied) = count_conflicts2(queens,r,queen[1])
                if c < min_value and (not occupied):
                    min_value = c
                    min_square = (r,queen[1])
                r+=1

            queens[q] = min_square
            #print_board(queens)
            #raw_input()
        else:
            conflict_free_iters += 1

        q = (q+1)%boardsize
        iters += 1

def validate(queens):
    conflicts = 0
    for q in queens:
        (c,o) = count_conflicts(queens,q[0],q[1])
        conflicts += c
    return conflicts == 0


