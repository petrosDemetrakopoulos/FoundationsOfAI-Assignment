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

        ################### Start of helper functions ###################
        def get_filled_row_values(row_index: int, state: GameState):
            """
            Returns the non-empty values of the row specified by a given row index.
            @param row_index: The row index
            @param state: The GameState object that describes the game in progress
            @return: A list containing the integer values of the specified row's non-empty cells
            """
            # returns non-empty values in row with index row_index
            filled_values = []
            for i in range(state.board.N):
                cur_cell = state.board.get(row_index, i)
                if cur_cell != SudokuBoard.empty:
                    filled_values.append(cur_cell)
            return filled_values

        def get_filled_column_values(column_index: int, state: GameState):
            """
            Returns the non-empty values of the column specified by a given column index.
            @param column_index: The column index
            @param state: The GameState object that describes the game in progress
            @return: A list containing the integer values of the specified column's non-empty cells.
            """
            filled_values = []
            for i in range(state.board.N):
                cur_cell = state.board.get(i, column_index)
                if cur_cell != SudokuBoard.empty:
                    filled_values.append(cur_cell)
            return filled_values

        def get_filled_region_values(row_index: int, column_index: int, state: GameState):
            """
            Returns the non-empty values of the (rectangular) region that the cell specified
            by the given row and column indices belongs to.
            @param row_index: The row index
            @param column_index: The column index
            @param state: The GameState object that describes the game in progress
            @return: A list containing the integer values of the specified region's non-empty cells
            """
            first_row = (row_index // state.board.m) * state.board.m
            # A smart way to determine the first row of the rectangular region where the cell belongs to,
            # is to get the integer part of the (row / m) fraction (floor division) and then multiply it by m.
            # The same logic is applied to determine the first column of the rectangular region in question.
            first_column = (column_index // state.board.n) * state.board.n
            filled_values = []
            # If first_row is the index of the first row of the region, then the index of the last row should be
            # first_row + state.board.m - 1
            for r in range(first_row, first_row + state.board.m):
                # If first_column is the index of the first column of the region, then the index of the last column
                # should be first_column + state.board.n - 1
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

        def evaluate_move_score_increase(state: GameState, move: Move):
            filled_row = get_filled_row_values(move.i, state)
            filled_col = get_filled_column_values(move.j, state)
            filled_block = get_filled_region_values(move.i, move.j, state)

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
                return 1
            # case where either a row, a column, a region or a combination of them can be immediately filled on the next move
            # and thus to easily provide points to the opponent
            # We want to introduce an artificial penalty (not reflected in the final score of the game) for proposing such moves
            # this will force the agent to avoid such moves as they allow the oponent to immediately score points afterwards
            elif len(filled_row) == full_len-1 or len(filled_col) == full_len-1 or len(filled_block) == full_len-1:
                is_row_almost_filled = len(filled_row) == full_len-1
                is_col_almost_filled = len(filled_col) == full_len-1
                is_block_almost_filled = len(filled_block) == full_len-1
                return -3*(is_row_almost_filled + is_col_almost_filled + is_block_almost_filled)
            return 0

        def possible(row_index, column_index, proposed_value):
            return game_state.board.get(row_index, column_index) == SudokuBoard.empty \
                   and not TabooMove(row_index, column_index, proposed_value) in game_state.taboo_moves

        def legal_moves_after_pruning(state, empty_cells):
            # prune any cell that we have no info about it (region, row and column containing it are empty)
            # the reasoning behind this pruning is that it is a bit naive to fill in cells for which we have no information and most probably there will be better options
            # this technique significantly reduces tree size and offers performance advantage
            # known_no_reward_cells list contains all empty cells except the ones that we have no info for
            known_no_reward_cells = []
            if not state.board.empty:
                for cell in empty_cells:
                    row_index = cell[0]
                    cell_index = cell[1]
                    if not (len(get_filled_row_values(row_index, state)) == 0 and
                            len(get_filled_column_values(cell_index, state)) == 0 and
                            len(get_filled_region_values(row_index, cell_index, state)) == 0):
                        known_no_reward_cells.append(cell)
            else:
                # in case of an empty board, we assign empty_cells to known_no_reward_cells
                # otherwise the pruning would prune all cells
                known_no_reward_cells = empty_cells

            # filter out illegal moves AND taboo moves from the known_no_reward_cells, 
            # the resulting list contains all moves which are both possible and LEGAL
            legal_moves = []
            for coords in known_no_reward_cells:
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

        # the function that initially triggers the recursion
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
                score_increase = evaluate_move_score_increase(state, legal_move)
                state.board.put(legal_move.i, legal_move.j,legal_move.value)
               
                if state.scores:
                    if state.scores[0]:
                        state.scores[0] += score_increase
                    else:
                        state.scores[0] = score_increase
                else:
                    state.scores = [0,score_increase]
                crn_max_score = minimax(state, max_depth, 0, -math.inf, math.inf, False)

                # clear legal_move from the board to continue by checking other possible moves (recurrsion unrolling)
                state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                # undo score increase
                state.scores[0] -= score_increase

                if crn_max_score > max_score:
                    best_move = Move(legal_move.i, legal_move.j, legal_move.value)
                    max_score = crn_max_score
            return best_move

        def minimax(state: GameState, max_depth: int, depth: int, alpha: float, beta: float, is_maximizing_player: bool):
            empty_cells = get_empty_cells(state)
            legal_moves = legal_moves_after_pruning(state, empty_cells)
            if depth >= max_depth: 
                # max depth reached, returning the score of the node
                return state.scores[0] - state.scores[1]
            
            if len(legal_moves) == 0:
                # no legal moves left, practically a leaf node
                # the evaluation function of a node is the difference between the score of the maximizer at this state and the score of the minimizer at the same state
                # this is the quantity that the minimax tries to maximize for maximizing player and minimize for the opponent
                return state.scores[0] - state.scores[1]

            if is_maximizing_player:
                # maximizer's move
                max_score = -math.inf
                
                for legal_move in legal_moves:
                    # calculate the amount by which the score of maximizing player will be increased if it plays legal_move
                    maximizer_score_increase = evaluate_move_score_increase(state, legal_move)
                    # play the move (add the move on the board)
                    state.board.put(legal_move.i, legal_move.j,legal_move.value)
                    # in the scores property of the game state we can find the scores of the maximizing and minimizing players
                    # because our logic is based on difference of scores after moves
                    # when plating a move we need to temporarily reflect its result on the score before continuing the search
                    if state.scores:
                        if state.scores[0]:
                            state.scores[0] += maximizer_score_increase
                        else:
                            state.scores[0] = maximizer_score_increase
                    else:
                        state.scores = [maximizer_score_increase, 0]
                    
                    # call minimax for the minimizing player
                    max_score = max(max_score, minimax(state, max_depth, depth+1, alpha, beta, False))

                    # undo the move
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                    # undo score increase
                    state.scores[0] -= maximizer_score_increase

                    # implementation of the alpha-beta prunning technique as demonstrated on
                    # https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning
                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break
                return max_score
            else:
                min_score = math.inf

                for legal_move in legal_moves:
                    minimizer_score_increase = evaluate_move_score_increase(state, legal_move)
                    state.board.put(legal_move.i, legal_move.j,legal_move.value)
                    # increase score for the maximizer in the current state
                    if state.scores:
                        if state.scores[1]:
                            state.scores[1] += minimizer_score_increase
                        else:
                            state.scores[1] = minimizer_score_increase
                    else:
                        state.scores = [0, minimizer_score_increase]

                    # call minimax for the maximizing player
                    min_score = min(min_score, minimax(state, max_depth, depth+1, alpha, beta, True))

                    # undo the move
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                    # undo score increase
                    state.scores[1] -= minimizer_score_increase

                    beta = min(beta, min_score)
                    if beta <= alpha:
                        break
                return min_score
        #################### End of helper functions ####################

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

        # initial depth 
        max_depth = 0
        while True:
            # iteratively increase tree depth to produce more accurate moves
            # on each iteration the move proposed should be slightly better than the move of the previous iteration
            max_depth += 1
            best_move = find_optimal_move(game_state, max_depth) # initial call to the recursive minimax function
            #print("minimax proposed move: " + str(best_move))
            if best_move != Move(-1, -1, -1):  # Failsafe mechanism to ensure we will never propose an invalid move
                if self.verbose:
                    # print statements for debug purposes
                    print("--------------")
                    print("Random move proposed: " + str(best_move))
                    print("Score for selected legal_move: " + str(evaluate_move_score_increase(game_state, best_move)))
                    print("Illegal moves for selected cell: " + str(get_illegal_moves(best_move.i, best_move.j, game_state)))
                    print("Block filled values for selected cell: " + str(get_filled_region_values(best_move.i, best_move.j, game_state)))
                    print("Row filled values for selected cell: " + str(get_filled_row_values(best_move.i, game_state)))
                    print("Column filled values for selected cell: " + str(get_filled_column_values(best_move.j, game_state)))
                    print("--------------")
                self.propose_move(best_move)
