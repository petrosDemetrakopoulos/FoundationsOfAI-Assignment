#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        # helper functions
        def get_row_filled_values(row_index: int, game_state: GameState):
            # returns non-empty values in row with index row_index 
            row = []
            for i in range(game_state.board.N):
                crn_cell = game_state.board.get(row_index, i)
                if crn_cell != 0: # 0 denotes an empty cell
                    row.append(crn_cell)
            return row

        def get_column_filled_values(column_index: int, state: GameState):
            # returns non-empty values in column with index column_index 
            column = []
            for i in range(state.board.N):
                crn_cell = state.board.get(i, column_index)
                if crn_cell != 0: # 0 denotes an empty cell
                    column.append(crn_cell)
            return column
        
        def get_block_filled_values(row: int, column: int, state:GameState):
            # return the non-empty values of the rectangular block that the cell with coordinates
            # (row, column) belongs to
            first_row = math.floor(row / state.board.m) *  state.board.m 
            # a smart way to find the first row of the rectangular region where the cell belongs to, 
            # is to round the row / m fraction to the lower closest integer and then multiply by m
            # we do the same to distinguish the first column of the rectangular region in question
            first_column = math.floor(column / state.board.n) *  state.board.n 
            rect = []
            for r in range(first_row, first_row + state.board.m):
                for c in range (first_column, first_column + state.board.n):
                    crn_cell = state.board.get(r, c)
                    if crn_cell != 0:
                        rect.append(crn_cell)
            return rect

        def illegal_moves(row: int, col: int, state:GameState):
            # return a list of numbers that CANNOT be added on a given empty cell (row,col)
            illegal = get_row_filled_values(row, state) + get_column_filled_values(col, state) + get_block_filled_values(row, col, state)
            return list(set(illegal)) # easy way to remove duplicates

        def score(move: Move, state: GameState):
            # return the score that should be added to a player after the given move
            # if the player has no score increase after the move, this function returns 0

            row = get_row_filled_values(move.i, state)
            col = get_column_filled_values(move.j, state)
            block = get_block_filled_values(move.i, move.j, state)
            full_len = state.board.N - 1

            # based onn the logic mentioned in the Assignment desctiption, we calculate score increase after the move
            if len(row) == full_len and len(col) == full_len and len(block) == full_len: # case where a row, a column and a block are completed after the move
                return 7
            elif len(row) == full_len and len(col) == full_len: # case where a row anda column is completed
                return 3
            elif len(row) == full_len and len(block) == full_len: #case where a row and a block is completed
                return 3
            elif len(col) == full_len and len(block) == full_len: #case where a col and a block is completed
                return 3
            elif len(row) == full_len or len(col) == full_len or len(block) == full_len: # case where only 1 among column, row and block are completed
                return 1
            else:
                return 0
        
        def get_empty_cells(state: GameState):
            empty_cells = []
            for i in range(len(state.board.squares)):
                if state.board.squares[i] == 0: # empty cell
                    empty_cells.append(state.board.f2rc(i))
            return empty_cells

            
        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in game_state.taboo_moves
        
        #filter out illegal moves AND taboo moves
        all_moves = [Move(i, j, value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value) and value not in illegal_moves(i, j, game_state)]

        # propose a valid move arbitrarily at first, then try to optimize it with minimax
        move = random.choice(all_moves)
        self.propose_move(move)


        while True:
            time.sleep(0.2)
            print("EMPTY CELLS")
            print(get_empty_cells(game_state))
            self.propose_move(random.choice(all_moves))



