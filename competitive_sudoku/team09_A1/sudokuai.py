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
    verbose = False  # a flag to print useful debug logs after each turn

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        # helper functions

        def get_row_filled_values(row_index: int, state: GameState):
            # returns non-empty values in row with index row_index
            row = []
            for i in range(state.board.N):
                crn_cell = state.board.get(row_index, i)
                if crn_cell != 0:  # 0 denotes an empty cell
                    row.append(crn_cell)
            return row

        def get_column_filled_values(column_index: int, state: GameState):
            # returns non-empty values in column with index column_index
            column = []
            for i in range(state.board.N):
                crn_cell = state.board.get(i, column_index)
                if crn_cell != 0:  # 0 denotes an empty cell
                    column.append(crn_cell)
            return column

        def get_block_filled_values(row: int, column: int, state: GameState):
            # return the non-empty values of the rectangular block that the cell with coordinates
            # (row, column) belongs to
            first_row = math.floor(row / state.board.m) * state.board.m
            # a smart way to find the first row of the rectangular region where the cell belongs to,
            # is to round the row / m fraction to the lower closest integer and then multiply by m
            # we do the same to distinguish the first column of the rectangular region in question
            first_column = math.floor(column / state.board.n) * state.board.n
            rect = []
            for r in range(first_row, first_row + state.board.m):
                for c in range(first_column, first_column + state.board.n):
                    crn_cell = state.board.get(r, c)
                    if crn_cell != 0:
                        rect.append(crn_cell)
            return rect

        def illegal_moves(row: int, col: int, state: GameState):
            # return a list of numbers that CANNOT be added on a given empty cell (row,col)
            illegal = get_row_filled_values(row, state) + get_column_filled_values(
                col, state) + get_block_filled_values(row, col, state)
            return set(illegal)  # easy way to remove duplicates

        def score_function(move: Move, state: GameState):
            # return the score that should be added to a player after the given move
            # if the player has no score increase after the move, this function returns 0
            row = get_row_filled_values(move.i, state)
            col = get_column_filled_values(move.j, state)
            block = get_block_filled_values(move.i, move.j, state)
            # N, excluding the cell under questiom (the cell we consider fo filling in)
            full_len = state.board.board_width() - 1

            # based onn the logic mentioned in the Assignment desctiption, we calculate score increase after the move
            # case where a row, a column and a block are completed after the move
            if len(row) == full_len and len(col) == full_len and len(block) == full_len:
                return 7
            # case where a row and a column is completed
            elif len(row) == full_len and len(col) == full_len:
                return 3
            # case where a row and a block is completed
            elif len(row) == full_len and len(block) == full_len:
                return 3
            # case where a col and a block is completed
            elif len(col) == full_len and len(block) == full_len:
                return 3
            # case where only 1 among column, row and block are completed
            elif len(row) == full_len or len(col) == full_len or len(block) == full_len:
                return 1
            else:
                return 0

        def get_empty_cells(state: GameState):
            # returns coordinates (as a tuple) of empty cells
            empty_cells = []
            for i in range(len(state.board.squares)):
                if state.board.squares[i] == 0:  # empty cell
                    empty_cells.append(state.board.f2rc(i))
            return empty_cells

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                and not TabooMove(i, j, value) in game_state.taboo_moves

        def minimax(state: GameState, isMaximizingPlayer: bool, crn_score: int):
            empty_cells_coords = []
            # compute empty cells coordinates
            for i in range(state.board.board_height()):
                for j in range(state.board.board_width()):
                    if state.board.get(i, j) == SudokuBoard.empty:
                        empty_cells_coords.append((i, j))

            if len(empty_cells_coords) == 0:
                # game end, all cells are filled
                return None, crn_score

            # prune any cell that we have no info about it (block, row and column containing it are empty)
            # still need to handle the issue with the empty board there
            # probably a simple iff will do
            cells_we_have_info_for = []
            for cell in empty_cells_coords: 
                cell_row = cell[0]
                cell_col = cell[1]
                if not (len(get_row_filled_values(cell_row, state)) == 0 and
                        len(get_column_filled_values(cell_col, state)) == 0 and
                        len(get_block_filled_values(cell_row, cell_col, state)) == 0):
                    cells_we_have_info_for.append(cell)

            # filter out illegal moves AND taboo moves from the empty_cells, these are all possible and legal moves
            all_legal_moves = [Move(coords[0], coords[1], value) for coords in cells_we_have_info_for for value in range(1, N+1)
                               if possible(coords[0], coords[1], value) and value not in illegal_moves(coords[0], coords[1], state)]
            print("legal moves len: " + str(len(all_legal_moves)))
            if len(all_legal_moves) == 0:  # no available legal move, so returning the latest legal move, probably triggered due to call of the minimax() after all cells have been fileld in the the original board
                print("NO LEGAL MOVES")
                return None, crn_score

            if isMaximizingPlayer:
                # initialize the crn_max_score with the minimum possible value supported by Python
                crn_max_score = -math.inf
                # arbitrariliy initialize optimal move to be the first legal move (in the loop we find the real optimal move)
                opt_move = all_legal_moves[0]
                for move in all_legal_moves:
                    new_score = score_function(move, state)
                    crn_score += new_score
                    state.board.put(move.i, move.j, move.value)
                    # calling now minimax for minimizing player
                    opt_move, max_score = minimax(state, False, crn_score)
                    # clear move from the board to continue by checking other possible moves
                    state.board.put(move.i, move.j, 0)
                    if max_score > crn_max_score:
                        crn_max_score = max_score
                        opt_move = move
                return opt_move, crn_max_score
            else:
                # initialize the crn_min_score with the maximum possible value supported by Python
                crn_min_score = math.inf
                # arbitrariliy initialize optimal move to be the first legal move (in the loop we find the real optimal move)
                opt_move = all_legal_moves[0]
                for move in all_legal_moves:
                    new_score = score_function(move, state)
                    crn_score -= new_score
                    state.board.put(move.i, move.j, move.value)
                    # calling now minimax for maximizing player
                    opt_move, min_score = minimax(state, True, crn_score)
                    # clear move from the board to continue by checking other possible moves
                    state.board.put(move.i, move.j, 0)
                    if min_score < crn_min_score:
                        crn_min_score = min_score
                        opt_move = move
                return opt_move, crn_min_score

        # filter out illegal moves AND taboo moves
        all_legal_moves = [Move(i, j, value) for i in range(N) for j in range(N)
                           for value in range(1, N+1) if possible(i, j, value) and value not in illegal_moves(i, j, game_state)]
        # propose a valid move arbitrarily at first, then try to optimize it with minimax and propose new moves as we still have time to do so
        move = random.choice(all_legal_moves)
        self.propose_move(move)

        while True:
            if self.verbose:
                print("--------------")
                print("Empty cells: " + str(get_empty_cells(game_state)))
                print("Score for selected move: " +
                      str(score(rndm_move, game_state)))
                print("Illegal moves for selected cell: " +
                      str(illegal_moves(rndm_move.i, rndm_move.j, game_state)))
                print("Block filled values for selected cell: " +
                      str(get_block_filled_values(rndm_move.i, rndm_move.j, game_state)))
                print("Row filled values for selected cell: " +
                      str(get_row_filled_values(rndm_move.i, game_state)))
                print("Column filled values for selected cell: " +
                      str(get_column_filled_values(rndm_move.j, game_state)))
                print("--------------")
            best_move, score = minimax(game_state, True, 0)
            print("returned move is: " + str(best_move))
            if best_move is not None:
                self.propose_move(best_move)
            time.sleep(1)
