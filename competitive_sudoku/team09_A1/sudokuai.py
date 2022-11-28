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

        def evaluate_state(state: GameState):
            for row in range(state.board.m):
                for col in range(state.board.n):
                    filled_row = get_filled_row_values(row, state)
                    filled_col = get_filled_column_values(col, state)
                    filled_block = get_filled_region_values(row, col, state)

                    full_len = N - 1
                    # based onn the logic mentioned in the assignment desctiption, we calculate score increase after the move
                    # case where a row, a column and a block are completed after the legal_move
                    if len(filled_row) == full_len and len(filled_col) == full_len and len(filled_block) == full_len:
                        return 7
                    # case where a row and a column is completed
                    elif len(filled_row) == full_len and len(filled_col) == full_len:
                        return 3
                    # case where a row and a block is completed
                    elif len(filled_row) == full_len and len(filled_block) == full_len:
                        return 3
                    # case where a col and a block is completed
                    elif len(filled_row) == full_len and len(filled_block) == full_len:
                        return 3
                    # case where only 1 among column, row and block are completed
                    elif len(filled_row) == full_len or len(filled_col) == full_len or len(filled_block) == full_len:
                        return 3
            return 0

        def possible(row_index, column_index, proposed_value):
            return game_state.board.get(row_index, column_index) == SudokuBoard.empty \
                   and not TabooMove(row_index, column_index, proposed_value) in game_state.taboo_moves

        def legal_moves_after_pruning(state, empty_cells):
            # prune any cell that we have no info about it (region, row and column containing it are empty)
            # the reasoning behind this pruning is that it is a bit naive to fill in cells for which we have no information and most probably there will be better options
            # this technique significantly reduces tree size and offers performance advantage
            # reward_cells list contain all empty cells except the ones that we have no info for
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
            # the resulting list contains all moves which are both possible and LEGAL
            legal_moves = []
            for coords in reward_cells:
                for value in range(1, N + 1):
                    if possible(coords[0], coords[1], value) and value not in get_illegal_moves(coords[0], coords[1], state):
                        legal_moves.append(Move(coords[0], coords[1], value))
            return legal_moves

        def get_empty_cells(state):
            empty_cells = []
            # compute empty cells coordinates
            # these are the cells that the agent can probably fill
            for i in range(N):
                for j in range(N):
                    if state.board.get(i, j) == SudokuBoard.empty:
                        empty_cells.append((i, j))
            return empty_cells

        def find_optimal_move(state, max_depth):
            max_score = -math.inf
            best_move = Move(-1,-1,-1)
            empty_cells = get_empty_cells(state)
            
            if len(empty_cells) == 0:
                # game end, all cells are filled, practically reached a leaf node
                return Move(-1, -1, -1)

            legal_moves = legal_moves_after_pruning(state, empty_cells)
           
            for legal_move in legal_moves:
                 # make the move
                state.board.put(legal_move.i, legal_move.j,legal_move.value)

                moveVal = minimax(state, max_depth, 0, -math.inf, math.inf, True)

                # clear legal_move from the board to continue by checking other possible moves
                state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                if moveVal > max_score:
                    best_move = Move(legal_move.i, legal_move.j, legal_move.value)
                    max_score = moveVal
            return best_move

        def minimax(state: GameState, max_depth: int, depth: int, alpha: float, beta: float, is_maximizing_player: bool):
            score = evaluate_state(state)
            if depth >= max_depth:
                return evaluate_state(state)

            empty_cells = get_empty_cells(state)

            legal_moves = legal_moves_after_pruning(state, empty_cells)
            if len(legal_moves) == 0:
                # no legal moves left
                return evaluate_state(state)
            # estimate a maximum depth we are willing to search up to
            # this maximum depth is a function of the length of the legal moves and the numbers of moves that have already been played. 
            # This depth limiting technique is used because we noticed that under circumstances (low time limit) 
            # the proposed moves were mostly random because the recursive minimax function did not have enough time to return the optimal move
            # the depth limit is given by the fraction (moves already played)/(legal moves the agent can play in the current state) multiplied by a constant
            # this quantity is monotonically increasing as the game progresses
            # Thus we force the algorithm to search deeper as the game progresses
            estimated_depth_limit = math.ceil(0.1*(len(game_state.moves) / len(legal_moves)))
            if depth >= estimated_depth_limit:
                return evaluate_state(state)
            
            if is_maximizing_player:
                # maximizer's move
                max_score = -math.inf

                for legal_move in legal_moves:
                    state.board.put(legal_move.i, legal_move.j,legal_move.value)
                    max_score = max(max_score, minimax(state, max_depth, depth+1, alpha, beta, not is_maximizing_player))

                    # undo the move
                    state.board.put(legal_move.i, legal_move.j,SudokuBoard.empty)

                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break
                return max_score
            else:
                min_score = math.inf

                for legal_move in legal_moves:
                    state.board.put(legal_move.i, legal_move.j,legal_move.value)

                    min_score = min(min_score, minimax(state, max_depth, depth+1, alpha, beta, not is_maximizing_player))

                    state.board.put(legal_move.i, legal_move.j,SudokuBoard.empty)

                    beta = min(beta, min_score)
                    if beta <= alpha:
                        break
                return min_score

        # compute_best_move body
        # filter out illegal moves AND taboo moves
        legal_moves = []
        for i in range(N):
            for j in range(N):
                for value in range(1, N + 1):
                    if possible(i, j, value) and value not in get_illegal_moves(i, j, game_state):
                       legal_moves.append(Move(i, j, value))

        # propose a valid move arbitrarily at first (random choice from legal moves), 
        # then try to optimize it with minimax and propose new moves as we still have time to do so
        move = random.choice(legal_moves)
        self.propose_move(move)
        print("random proposed:" + str(move))

        # initial depth 
        max_depth = 0
        while True:
            # iteratively increase tree depth to produce more accurate moves
            # on each iteration the move proposed should be slightly better than the move of the previous iteration
            max_depth += 1
            print("max_depth = " + str(max_depth))
            best_move = find_optimal_move(game_state, max_depth) # initial call to the recursive minimax function
            print("minimax proposed move: " + str(best_move))
            if best_move != Move(-1, -1, -1):  # Failsafe mechanism to ensure we will never propose an invalid move
                if self.verbose:
                    # print statements for debug purposes
                    print("--------------")
                    print("Random move proposed: " + str(best_move))
                    print("Score for selected legal_move: " + str(evaluate(best_move, game_state)))
                    print("Illegal moves for selected cell: " + str(get_illegal_moves(best_move.i, best_move.j, game_state)))
                    print("Block filled values for selected cell: " + str(get_filled_region_values(best_move.i, best_move.j, game_state)))
                    print("Row filled values for selected cell: " + str(get_filled_row_values(best_move.i, game_state)))
                    print("Column filled values for selected cell: " + str(get_filled_column_values(best_move.j, game_state)))
                    print("--------------")
                self.propose_move(best_move)
