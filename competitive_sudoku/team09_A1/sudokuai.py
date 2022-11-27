#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
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

    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # helper functions
        def get_filled_row_values(row_index: int, state: GameState):
            # returns non-empty values in row with index row_index
            filled_values = []
            for i in range(state.board.N):
                cur_cell = state.board.get(row_index, i)
                if cur_cell != SudokuBoard.empty:
                    filled_values.append(cur_cell)
            return filled_values

        def get_filled_column_values(column_index: int, state: GameState):
            # returns non-empty values in column with index column_index
            filled_values = []
            for i in range(state.board.N):
                cur_cell = state.board.get(i, column_index)
                if cur_cell != SudokuBoard.empty:
                    filled_values.append(cur_cell)
            return filled_values

        def get_filled_region_values(row_index: int, column_index: int, state: GameState):
            # return the non-empty values of the rectangular block that the cell with coordinates
            # (row, column) belongs to
            first_row = math.floor(row_index / state.board.m) * state.board.m
            # a smart way to find the first row of the rectangular region where the cell belongs to,
            # is to round the row / m fraction to the lower closest integer (floor) and then multiply by m
            # we do the same to distinguish the first column of the rectangular region in question respectively
            first_column = math.floor(column_index / state.board.n) * state.board.n
            filled_values = []
            for r in range(first_row, first_row + state.board.m):
                for c in range(first_column, first_column + state.board.n):
                    crn_cell = state.board.get(r, c)
                    if crn_cell != SudokuBoard.empty:
                        filled_values.append(crn_cell)
            return filled_values

        def get_illegal_moves(row_index: int, col_index: int, state: GameState):
            # return a list of numbers that CANNOT be added on a given empty cell (row,col)
            # these numbers are illegal and the cannot be set to the given cell they already exist in at least 1 out of the cell row, column or block
            illegal = get_filled_row_values(row_index, state) + get_filled_column_values(col_index, state) + get_filled_region_values(row_index, col_index, state)
            return set(illegal)  # easy way to remove duplicates 

        def score_function(move: Move, state: GameState):
            # return the score that should be added to a player after the given legal_move
            # if the player has no score increase after the legal_move, this function returns 0
            # this is the most obvious scoring function for the nodes as it perfectly expresses the objective of the player
            row = get_filled_row_values(move.i, state)
            col = get_filled_column_values(move.j, state)
            block = get_filled_region_values(move.i, move.j, state)
            # N, excluding the cell under question (the cell we consider fo filling in)
            # we exclude it because scoring function is called BEFORE the cell gets filled
            full_len = N - 1

            # based onn the logic mentioned in the Assignment desctiption, we calculate score increase after the legal_move
            # case where a row, a column and a block are completed after the legal_move
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

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                and not TabooMove(i, j, value) in game_state.taboo_moves

        def estimate_depth_limit(legal_moves_len: int):
            # c is a "conservativeness" factor
            # we empirically figured out that a value that works pretty well for c is 1.8
            # we raise this constant to the logarith of the number of moves that have already been played
            # we use the logarithm to limit the exponential growth of the returned value
            c = 1.8
            return int(c**(math.log(len(game_state.moves))))

        def minimax(state: GameState, depth: int, alpha: float, beta: float, is_maximizing_player: bool, cur_score: int):
            empty_cells = []
            # compute empty cells coordinates
            for i in range(N):
                for j in range(N):
                    if state.board.get(i, j) == SudokuBoard.empty:
                        empty_cells.append((i, j))

            if len(empty_cells) == 0:
                # game end, all cells are filled, practically reached a leaf node
                return None, cur_score

            # prune any cell that we have no info about it (region, row and column containing it are empty)
            # the reasoning behind it is that it is a bit naive to fill in cells for which we have not information and most probably there will be better options
            # this technique significantly reduces tree size and offers performance advantage
            reward_cells = []
            if not state.board.empty:
                for cell in empty_cells:
                    row_index = cell[0]
                    cell_index = cell[1]
                    if not (len(get_filled_row_values(row_index, state)) == 0 and
                            len(get_filled_column_values(cell_index, state)) == 0 and
                            len(get_filled_region_values(row_index, cell_index, state)) == 0):
                        reward_cells.append(cell)
            else:
                # in case of an empty board, we assign empty_cells to reward_cells
                # otherwise the pruning would prune all cells
                reward_cells = empty_cells

            # filter out illegal moves AND taboo moves from the empty_cells, 
            # the resulting list contains all possible and legal moves
            legal_moves = []
            for coords in reward_cells:
                for value in range(1, N + 1):
                    if possible(coords[0], coords[1], value) and value not in get_illegal_moves(coords[0],coords[1],state):
                        legal_moves.append(Move(coords[0], coords[1], value))

            # no available legal_move, probably triggered due to call of the minimax() after all cells have been fileld in the the original board
            if len(legal_moves) == 0:
                if self.verbose:
                    print("No legal moves left")
                return None, cur_score

            # estimate a maximum depth we are willing to search up to
            # this maximum depth is a function of the length of the legal moves. more on it on the report
            estimated_depth_limit = estimate_depth_limit(len(legal_moves))
            if depth > estimated_depth_limit:
                return None, cur_score

            if is_maximizing_player:
                # initialize the cur_max_score with the minimum possible value supported by Python
                cur_max_score = -math.inf
                # arbitrariliy initialize optimal legal_move to be the first legal legal_move (in the loop we find the real optimal legal_move)
                opt_move = legal_moves[0]
                for legal_move in legal_moves:
                    new_score = score_function(legal_move, state)
                    cur_score += new_score
                    state.board.put(legal_move.i, legal_move.j,legal_move.value)
                    # calling now minimax for minimizing player
                    opt_move, max_score = minimax(state, depth + 1, alpha, beta, False, cur_score)

                    # clear legal_move from the board to continue by checking other possible moves
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)
                    if max_score > cur_max_score:
                        opt_move = legal_move
                    cur_max_score = max(cur_max_score, max_score)

                    # alpha-beta pruning
                    # implemented as suggested in https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
                    alpha = max(alpha, cur_max_score)
                    if beta <= alpha:
                        break

                return opt_move, cur_max_score
            else:
                # initialize the cur_min_score with the maximum possible value supported by Python
                cur_min_score = math.inf
                # arbitrariliy initialize optimal legal_move to be the first legal legal_move (in the loop we find the real optimal legal_move)
                opt_move = legal_moves[0]
                for legal_move in legal_moves:
                    new_score = score_function(legal_move, state)
                    cur_score -= new_score
                    state.board.put(legal_move.i, legal_move.j,
                                    legal_move.value)
                    # calling now minimax for maximizing player
                    opt_move, min_score = minimax(state, depth + 1, alpha, beta, True, cur_score)

                    # clear legal_move from the board to continue by checking other possible moves
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)
                    if min_score < cur_min_score:
                        opt_move = legal_move
                    cur_min_score = min(cur_min_score, min_score)

                    # alpha-beta pruning
                    beta = min(beta, cur_min_score)
                    if beta <= alpha:
                        break

                return opt_move, cur_min_score

        # filter out illegal moves AND taboo moves
        legal_moves = []
        for i in range(N):
            for j in range(N):
                for value in range(1, N + 1):
                    if possible(i, j, value) and value not in get_illegal_moves(i, j, game_state):
                       legal_moves.append(Move(i, j, value))

        # propose a valid legal_move arbitrarily at first, 
        # then try to optimize it with minimax and propose new moves as we still have time to do so
        move = random.choice(legal_moves)
        self.propose_move(move)

        if self.verbose:
            # print statements for debug purposes
            print("--------------")
            print("Random move proposed: " + str(move))
            print("Score for selected legal_move: " +
                  str(score_function(move, game_state)))
            print("Illegal moves for selected cell: " +
                  str(get_illegal_moves(move.i, move.j, game_state)))
            print("Block filled values for selected cell: " +
                  str(get_filled_region_values(move.i, move.j, game_state)))
            print("Row filled values for selected cell: " +
                  str(get_filled_row_values(move.i, game_state)))
            print("Column filled values for selected cell: " +
                  str(get_filled_column_values(move.j, game_state)))
            print("--------------")

        best_move, score = minimax(game_state, 0, -math.inf, math.inf, True, 0)
        if best_move is not None:  # Failsafe mechanism to ensure we will never propose a None move
            self.propose_move(best_move)
