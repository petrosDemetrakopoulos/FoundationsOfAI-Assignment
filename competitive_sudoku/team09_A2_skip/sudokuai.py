#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    verbose = False  # a flag to print useful debug logs after each turn

    def __init__(self):
        super().__init__()
        self.move_skipped = False
        self.N = -1
        self.range_N = range(self.N)
        self.range_N_plus_1 = range(1, self.N + 1)

    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        range_N = range(N)
        range_N_plus_1 = range(1,N+1)

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
            start = time.time()
            for i in range_N:
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
            for i in range_N:
                cur_cell = state.board.get(i, column_index)
                if cur_cell != SudokuBoard.empty:
                    filled_values.append(cur_cell)
            return filled_values

        def get_filled_block_values(row_index: int, column_index: int, state: GameState):
            """
            Returns the non-empty values of the (rectangular) block that the cell specified
            by the given row and column indices belongs to.
            @param row_index: The row index
            @param column_index: The column index
            @param state: The GameState object that describes the game in progress
            @return: A list containing the integer values of the specified block's non-empty cells
            """
            first_row = (row_index // state.board.m) * state.board.m
            # A smart way to determine the first row of the rectangular block where the cell belongs to,
            # is to get the integer part of the (row / m) fraction (floor division) and then multiply it by m.
            # The same logic is applied to determine the first column of the rectangular block in question.
            first_column = (column_index // state.board.n) * state.board.n
            filled_values = []
            # If first_row is the index of the first row of the block, then the index of the last row should be
            # first_row + state.board.m - 1
            for r in range(first_row, first_row + state.board.m):
                # If first_column is the index of the first column of the block, then the index of the last column
                # should be first_column + state.board.n - 1
                for c in range(first_column, first_column + state.board.n):
                    crn_cell = state.board.get(r, c)
                    if crn_cell != SudokuBoard.empty:
                        filled_values.append(crn_cell)
            return filled_values

        def get_illegal_moves(row_index: int, col_index: int, state: GameState):
            """
            Returns a list of numbers that already exist in the specified cell's row, column or block. These numbers
            are illegal values and CANNOT be put on the given empty cell.
            @param row_index: The empty cell's row index
            @param col_index: The empty cell's column index
            @param state: The GameState object that describes the game in progress
            @return: A list of integers representing the illegal values of the specified empty cell.
            """
            illegal = get_filled_row_values(row_index, state) + get_filled_column_values(col_index, state) + get_filled_block_values(row_index, col_index, state)
            return set(illegal)  # Easy way to remove duplicates

        def evaluate_move_score_increase(move: Move, state: GameState, allow_recusion=True):
            """
            Calculates the score increase achieved after the proposed move is made.
            @param move: A Move object that describes the proposed move
            @param state: The GameState object that describes the game in progress
            @return: The calculated score increase achieved by the proposed move
            """
            filled_row = get_filled_row_values(move.i, state)
            filled_col = get_filled_column_values(move.j, state)
            filled_block = get_filled_block_values(move.i, move.j, state)

            full_len = N - 1
            score = 0
            # Case where a row, a column and a block are completed after the proposed move is made
            if len(filled_row) == full_len and len(filled_col) == full_len and len(filled_block) == full_len:
                score = 7
            # Case where a row and a column are completed after the proposed move is made
            elif len(filled_row) == full_len and len(filled_col) == full_len:
                score = 3
            # Case where a row and a block are completed after the proposed move is made
            elif len(filled_row) == full_len and len(filled_block) == full_len:
                score = 3
            # Case where a col and a block are completed after the proposed move is made
            elif len(filled_row) == full_len and len(filled_block) == full_len:
                score = 3
            # Case where only 1 among column, row and block are completed after the proposed move is made
            elif len(filled_row) == full_len or len(filled_col) == full_len or len(filled_block) == full_len:
                score =  1
            
            # Case where either a row, a column, a block or a combination of them can be immediately filled during the
            # next game turn, thus easily providing points to the opponent. Our intention is to introduce an artificial
            # "penalty" (not reflected in the final score of the game) for the proposal of such moves. This will force
            # the agent to avoid such moves, as they allow the opponent to immediately score points afterwards.
            is_row_almost_filled = len(filled_row) == full_len-1
            is_col_almost_filled = len(filled_col) == full_len-1
            is_block_almost_filled = len(filled_block) == full_len-1

            # The allow recurstion parameter is needed so that we don't get stuck in an infinite loop.
            # Essentially what this says is to not consider these rules when this function is called with the
            # purpose of obtaining a score with which we decrease a score of the move we initially wanted to score
            if allow_recusion:
                if is_row_almost_filled:
                    # Get missing value in row
                    missing_value = list(set(range(1,full_len+2)) - set(filled_row))[0]
                    empty_cell_index = [x for x in get_empty_cells(state) if x[0] == move.i][0]
                    # Place move that is immediately available for point scoring
                    state.board.put(empty_cell_index[0], empty_cell_index[1], missing_value)
                    # Evaluate that move to check how many points it awards
                    move_score = evaluate_move_score_increase(Move(empty_cell_index[0], empty_cell_index[1], missing_value), state, False)
                    potential_row_move_points_lost = move_score
                    # Remove move from board to return to original state
                    state.board.put(empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
                else:
                    potential_row_move_points_lost = 0

                if is_col_almost_filled:
                    # Get missing value in column
                    missing_value = list(set(range(1,full_len+2)) - set(filled_col))[0]
                    empty_cell_index = [x for x in get_empty_cells(state) if x[1] == move.j][0]
                    # Place move that is immediately available for point scoring
                    state.board.put(empty_cell_index[0], empty_cell_index[1], missing_value)
                    # Evaluate that move to check how many points it awards
                    move_score = evaluate_move_score_increase(Move(empty_cell_index[0], empty_cell_index[1], missing_value), state, False)
                    potential_col_move_points_lost = move_score
                    # Remove move from board to return to original state
                    state.board.put(empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
                else:
                    potential_col_move_points_lost = 0

                if is_block_almost_filled:
                    # Get missing value in block
                    missing_value = list(set(range(1,full_len+2)) - set(filled_block))[0]
                    
                    first_row = (move.i // state.board.m) * state.board.m
                    first_column = (move.j // state.board.n) * state.board.n
                    empty_cell_index = [x for x in get_empty_cells(state) if x[0] in range(first_row, first_row + state.board.m) and x[1] in range(first_column, first_column + state.board.n)][0]
                    # Place move that is immediately available for point scoring
                    state.board.put(empty_cell_index[0], empty_cell_index[1], missing_value)
                    # Evaluate that move to check how many points it awards
                    move_score = evaluate_move_score_increase(Move(empty_cell_index[0], empty_cell_index[1], missing_value), state, False)
                    potential_block_move_points_lost = move_score
                    # Remove move from board to return to original state
                    state.board.put(empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
                else:
                    potential_block_move_points_lost = 0

                score = score - max(potential_row_move_points_lost, potential_col_move_points_lost, potential_block_move_points_lost)
                
            return score

        def possible(row_index, column_index, proposed_value):
            return game_state.board.get(row_index, column_index) == SudokuBoard.empty \
                   and not TabooMove(row_index, column_index, proposed_value) in game_state.taboo_moves

        def legal_moves_after_pruning(state, empty_cells):
            # prune any cell that we have no info about it (block, row and column containing it are empty)
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
                            len(get_filled_block_values(row_index, cell_index, state)) == 0):
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
            """
            Returns the empty cells of the sudoku board at a specified game state
            @param state: The GameState object that describes the current state of the game in progress
            @return: A list of integer tuples (i, j) representing the coordinates of the empty cells
            present in the Sudoku board at its current game state
            """
            # Compute empty cells coordinates
            # These are the cells that the agent can probably fill
            empty_cells = [(i,j) for i in range_N for j in range_N if state.board.get(i, j) == SudokuBoard.empty]
            return empty_cells

        # the function that initially triggers the recursion
        def find_optimal_move(state, max_depth):
            """
            Used as a helper function that triggers Minimax's recursive call
            @param state: The GameState object that describes the current state of the game in progress
            @param max_depth: The maximum depth to be reached by Minimax's tree
            @return: A Move object representing the best game move determined through Minimax's recursion
            """
             # Initialize max_score with the lowest possible supported value
            max_score = -math.inf
            # find all empty cells
            empty_cells = get_empty_cells(state)

            if len(empty_cells) == 0:
                # game end, all cells are filled, practically reached a leaf node
                return Move(-1, -1, -1)

            # initialize best_move to an invalid move 
            best_move = Move(-1,-1,-1)
            # find all possible legal moves for the current game state
            legal_moves = legal_moves_after_pruning(state, empty_cells)

            for legal_move in legal_moves:
                # Calculate the amount by which the score of the maximizing player will be increased if it plays legal_move
                score_increase = evaluate_move_score_increase(legal_move, state)
                # Make the move
                state.board.put(legal_move.i, legal_move.j, legal_move.value)

                # Increase the score of the player at the current state.
                # The score of the maximizing player is saved at state.scores[0] and 
                # the score of minimizing player is saved at state.scores[1]
                if state.scores:
                    if state.scores[0]:
                        state.scores[0] += score_increase
                    else:
                        state.scores[0] = score_increase
                else:
                    state.scores = [0, score_increase]
                cur_max_score = minimax(state, max_depth, 0, -math.inf, math.inf, False)

                # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
                state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                # Undo the score increase to continue by checking other possible moves (recursion unrolling)
                state.scores[0] -= score_increase

                if cur_max_score > max_score:
                    best_move = Move(legal_move.i, legal_move.j, legal_move.value)
                    max_score = cur_max_score

            return best_move
        
        def get_skip_move(legal_moves: list, game_state: GameState):
            print("DEBA")
            # iterate rows to find potential move than can force the agent to "skip" the move
            
            for i in range(game_state.board.N):
                print("deba")
                available_moves_in_row = [move for move in legal_moves if move.i == i]
                available_cells_in_row = list(set([move.j for move in available_moves_in_row]))
                moves_per_cell = {col_index : [] for col_index in available_cells_in_row}
                for move in available_moves_in_row:
                    moves_per_cell[move.j].append(move.value)
                print(moves_per_cell)
                non_ambiguous_value = None
                for col_index, moves_list in moves_per_cell.items():
                    if len(moves_list) == 1:
                        non_ambiguous_value = moves_list[0]

                if non_ambiguous_value is not None:
                    # If a cell in the row can contain only a single value
                    # but another cell from the row can also have it
                    # then it means that putting it in the latter one will result in an unsolvable sudoku
                    # we want to propose that move in order to "skip" the turn
                    for col_index, moves_list in moves_per_cell.items():
                        if len(moves_list) > 1 and non_ambiguous_value in moves_list:
                            return Move(i, col_index, non_ambiguous_value)  
            for j in range(0,game_state.board.N):
                available_moves_in_col = [move for move in legal_moves if move.j == j]
                available_cells_in_col = list(set([move.i for move in available_moves_in_col]))
                moves_per_cell = {row_index : [] for row_index in available_cells_in_col}
                for move in available_moves_in_col:
                    moves_per_cell[move.i].append(move.value)
                print(moves_per_cell)
                non_ambiguous_value = None
                for row_index, moves_list in moves_per_cell.items():
                    if len(moves_list) == 1:
                        non_ambiguous_value = moves_list[0]

                if non_ambiguous_value is not None:
                    # If a cell in the row can contain only a single value
                    # but another cell from the row can also have it
                    # then it means that putting it in the latter one will result in an unsolvable sudoku
                    # we want to propose that move in order to "skip" the turn
                    for row_index, moves_list in moves_per_cell.items():
                        if len(moves_list) > 1 and non_ambiguous_value in moves_list:
                            return Move(row_index, j, non_ambiguous_value)
            
            print("OSHTE PO DEBA")
            return None

        def get_greedy_move(state: GameState, legal_moves):
            max_move = legal_moves[0]
            max_score = -1
            for move in legal_moves:
                max_eval = evaluate_move_score_increase(move, state)
                if max_eval > max_score:
                    max_move = move
                    max_score = max_eval
            return max_move

        def minimax(state: GameState, max_depth: int, depth: int, alpha: float, beta: float, is_maximizing_player: bool):
            """
            Implementation of the Minimax algorithm that includes alpha-beta pruning
            @param state: The GameState object that describes the current state of the game in progress
            @param max_depth: The maximum depth to be reached by Minimax's tree
            @param depth: The current depth reached by Minimax's tree
            @param alpha: The alpha value (used for alpha-beta pruning)
            @param beta: The beta value (used for alpha-beta pruning)
            @param is_maximizing_player: A boolean flag indicating whether it is the maximizing player's turn to play
            @return: The maximum maximizer-minimizer score difference achieved by the Minimax Algorithm
            """
            empty_cells = get_empty_cells(state)
            # find out any legal moves we can do at the current game state
            legal_moves = legal_moves_after_pruning(state, empty_cells)

            if depth >= max_depth: 
                # Max depth reached, returning the score of the node
                return state.scores[0] - state.scores[1]
            
            if len(legal_moves) == 0:
                # No legal moves left, practically a leaf node The evaluation function of a node is the difference
                # between the score of the maximizer at this state and the score of the minimizer at the same state.
                # This is the quantity that the minimax tries to maximize for the maximizing player and minimize for the
                # opponent
                return state.scores[0] - state.scores[1]

            if is_maximizing_player:
                # Maximizer's move
                # Initialize max_score with the lowest possible supported value
                max_score = -math.inf
                
                for legal_move in legal_moves:
                    # Calculate the amount by which the score of the maximizing player will increase if it plays
                    # legal_move
                    maximizer_score_increase = evaluate_move_score_increase(legal_move, state)

                    # Play the move (add the move to the sudoku board)
                    state.board.put(legal_move.i, legal_move.j, legal_move.value)

                    # In the "scores" property of the GameState object we can find the scores of the maximizing and
                    # minimizing players. Since our logic is based on the difference of scores between the maximizing
                    # player and minimizing player after a move is played, we need to temporarily reflect the move's
                    # result on the player score before continuing the search
                    if state.scores:
                        if state.scores[0]:
                            state.scores[0] += maximizer_score_increase
                        else:
                            state.scores[0] = maximizer_score_increase
                    else:
                        state.scores = [maximizer_score_increase, 0]
                    
                    # Call minimax for the minimizing player
                    max_score = max(max_score, minimax(state, max_depth, depth+1, alpha, beta, False))

                    # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                    # Undo the score increase to continue by checking other possible moves (recursion unrolling)
                    state.scores[0] -= maximizer_score_increase

                    # Implementation of the alpha-beta pruning technique as demonstrated on
                    # https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning
                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break

                return max_score
            else:
                # minimizer's move
                # initialize min_score with the highest possible supported value
                min_score = math.inf

                for legal_move in legal_moves:
                    # Calculate the amount by which the score of the minimizing player will increase if it plays
                    # legal_move
                    minimizer_score_increase = evaluate_move_score_increase(legal_move, state)

                    # Play the move (add the move to the sudoku board)
                    state.board.put(legal_move.i, legal_move.j, legal_move.value)

                    # Increase score for the maximizer in the current state
                    if state.scores:
                        if state.scores[1]:
                            state.scores[1] += minimizer_score_increase
                        else:
                            state.scores[1] = minimizer_score_increase
                    else:
                        state.scores = [0, minimizer_score_increase]

                    # Call minimax for the maximizing player
                    min_score = min(min_score, minimax(state, max_depth, depth+1, alpha, beta, True))

                    # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
                    state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

                    # Undo the score increase to continue by checking other possible moves (recursion unrolling)
                    state.scores[1] -= minimizer_score_increase

                    beta = min(beta, min_score)
                    if beta <= alpha:
                        break

                return min_score
        #################### End of helper functions ####################

        # Filter out illegal moves AND taboo moves
        legal_moves = [Move(i,j,value)  for i in range_N for j in range_N  for value in range_N_plus_1 if possible(i, j, value) and value not in get_illegal_moves(i, j, game_state)]
        # Propose a valid move arbitrarily at first (random choice from legal moves),
        # then keep finding optimal moves with minimax and propose them for as long as we are given the time to do so.
        rndm_move = random.choice(legal_moves)
        self.propose_move(rndm_move)
        move = get_greedy_move(game_state, legal_moves)
        self.propose_move(move)

        empty_cells_count = len(get_empty_cells(game_state))
        # If we are not on track to make the last move
        # attempt to skip the move by proposing a move that would make the sudoku unsolvable
        if empty_cells_count % 2 == 0:
            skip_move = get_skip_move(legal_moves, game_state)
            if skip_move is not None:
                self.propose_move(skip_move)
                self.move_skipped = True
        if not self.move_skipped:
            # Initial Minimax search depth
            max_depth = 0
            while True:
                # Iteratively increase Minimax's tree depth to discover more optimal moves.
                # On each iteration the move proposed should be slightly better than the move of the previous iteration.
                max_depth += 1
                best_move = find_optimal_move(game_state, max_depth)  # Initial call to the recursive minimax function
                if best_move != Move(-1, -1, -1):  # Failsafe mechanism to ensure we will never propose an invalid move
                    if self.verbose:
                        # Print statements for debug purposes
                        print("--------------")
                        print("Random move proposed: " + str(best_move))
                        print("Score for selected legal_move: " + str(evaluate_move_score_increase(best_move, game_state)))
                        print("Illegal moves for selected cell: " + str(get_illegal_moves(best_move.i, best_move.j, game_state)))
                        print("Block filled values for selected cell: " + str(get_filled_block_values(best_move.i, best_move.j, game_state)))
                        print("Row filled values for selected cell: " + str(get_filled_row_values(best_move.i, game_state)))
                        print("Column filled values for selected cell: " + str(get_filled_column_values(best_move.j, game_state)))
                        print("--------------")
                    self.propose_move(best_move)
