#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import time


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    verbose = False  # a flag to print useful debug logs after each turn

    def __init__(self):
        super().__init__()
        self.N = -1
        self.range_N = range(self.N)
        self.range_N_plus_1 = range(1, self.N + 1)

    ###### Start of helper functions ######
    def get_filled_row_values(self, row_index: int, game_state: GameState):
        """
        Returns the non-empty values of the row specified by a given row index.
        @param row_index: The row index
        @param game_state: The GameState object that describes the game in progress
        @return: A list containing the integer values of the specified row's non-empty cells
        """
        # returns non-empty values in row with index row_index
        filled_values = []
        for i in self.range_N:
            cur_cell = game_state.board.get(row_index, i)
            if cur_cell != SudokuBoard.empty:
                filled_values.append(cur_cell)
        return filled_values

    def get_filled_column_values(self, column_index: int, game_state: GameState):
        """
        Returns the non-empty values of the column specified by a given column index.
        @param column_index: The column index
        @param game_state: The GameState object that describes the game in progress
        @return: A list containing the integer values of the specified column's non-empty cells.
        """
        filled_values = []
        for i in self.range_N:
            cur_cell = game_state.board.get(i, column_index)
            if cur_cell != SudokuBoard.empty:
                filled_values.append(cur_cell)
        return filled_values

    def get_filled_block_values(self, row_index: int, column_index: int, game_state: GameState):
        """
        Returns the non-empty values of the (rectangular) block that the cell specified
        by the given row and column indices belongs to.
        @param row_index: The row index
        @param column_index: The column index
        @param game_state: The GameState object that describes the game in progress
        @return: A list containing the integer values of the specified block's non-empty cells
        """
        first_row = (row_index // game_state.board.m) * game_state.board.m
        # A smart way to determine the first row of the rectangular block where the cell belongs to,
        # is to get the integer part of the (row / m) fraction (floor division) and then multiply it by m.
        # The same logic is applied to determine the first column of the rectangular block in question.
        first_column = (column_index // game_state.board.n) * \
            game_state.board.n
        filled_values = []
        # If first_row is the index of the first row of the block, then the index of the last row should be
        # first_row + game_state.board.m - 1
        for r in range(first_row, first_row + game_state.board.m):
            # If first_column is the index of the first column of the block, then the index of the last column
            # should be first_column + game_state.board.n - 1
            for c in range(first_column, first_column + game_state.board.n):
                crn_cell = game_state.board.get(r, c)
                if crn_cell != SudokuBoard.empty:
                    filled_values.append(crn_cell)
        return filled_values

    def get_illegal_moves(self, row_index: int, col_index: int, game_state: GameState):
        """
        Returns a list of numbers that already exist in the specified cell's row, column or block. These numbers
        are illegal values and CANNOT be put on the given empty cell.
        @param row_index: The empty cell's row index
        @param col_index: The empty cell's column index
        @param game_state: The GameState object that describes the game in progress
        @return: A list of integers representing the illegal values of the specified empty cell.
        """
        illegal = self.get_filled_row_values(row_index, game_state) + self.get_filled_column_values(col_index,
                                                                                                    game_state) + self.get_filled_block_values(
            row_index, col_index, game_state)
        return set(illegal)  # Easy way to remove duplicates

    def evaluate_move_score_increase(self, move: Move, game_state: GameState, allow_recusion=True):
        """
        Calculates the score increase achieved after the proposed move is made.
        @param move: A Move object that describes the proposed move
        @param game_state: The GameState object that describes the game in progress
        @return: The calculated score increase achieved by the proposed move
        """
        filled_row = self.get_filled_row_values(move.i, game_state)
        filled_col = self.get_filled_row_values(move.j, game_state)
        filled_block = self.get_filled_block_values(move.i, move.j, game_state)

        full_len = self.N - 1
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
            score = 1

        # Case where either a row, a column, a block or a combination of them can be immediately filled during the
        # next game turn, thus easily providing points to the opponent. Our intention is to introduce an artificial
        # "penalty" (not reflected in the final score of the game) for the proposal of such moves. This will force
        # the agent to avoid such moves, as they allow the opponent to immediately score points afterwards.
        is_row_almost_filled = len(filled_row) == full_len - 1
        is_col_almost_filled = len(filled_col) == full_len - 1
        is_block_almost_filled = len(filled_block) == full_len - 1
        # The "allow_recurstion" parameter is used to avoid getting stuck in an infinite loop.
        # This check of the allow recursion parameter is needed because when we evaluate our own moves
        # and want to reason about the points the opponent can score with the next move
        # we want to keep the heuristic out of the calculation and get the actual score the opponent can get
        if allow_recusion:
            full_len_range = range(1, full_len + 2)
            empty_cells = self.get_empty_cells(game_state)
            if is_row_almost_filled:
                # Calculate which value is missing from the row under examination
                # We do so by finding the difference between the sets containing all the nxm values that must be present in a complete row
                # and the set containing the values that are currently filled in the row
                # we follow the same reasoning for columns and blocks in the following if statements
                missing_value = list(set(full_len_range) - set(filled_row))[0]
                empty_cell_index = [
                    x for x in empty_cells if x[0] == move.i][0]
                # Place move that is immediately available for point scoring
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], missing_value)
                # Evaluate that move to check how many points it awards
                move_score = self.evaluate_move_score_increase(
                    Move(empty_cell_index[0], empty_cell_index[1], missing_value), game_state, False)
                potential_row_move_points_lost = move_score
                # Remove move from board to return to original game_state
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
            else:
                potential_row_move_points_lost = 0

            if is_col_almost_filled:
                # Calculate which value is missing from the collumn under examinationive
                missing_value = list(set(full_len_range) - set(filled_col))[0]
                empty_cell_index = [
                    x for x in empty_cells if x[1] == move.j][0]
                # Place move that is immediately available for point scoring
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], missing_value)
                # Evaluate that move to check how many points it awards
                move_score = self.evaluate_move_score_increase(
                    Move(empty_cell_index[0], empty_cell_index[1], missing_value), game_state, False)
                potential_col_move_points_lost = move_score
                # Remove move from board to return to original game_state
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
            else:
                potential_col_move_points_lost = 0

            if is_block_almost_filled:
                # Calculate which value is missing from the column under examination
                missing_value = list(
                    set(full_len_range) - set(filled_block))[0]

                first_row = (move.i // game_state.board.m) * game_state.board.m
                first_column = (move.j // game_state.board.n) * \
                    game_state.board.n
                empty_cell_index = [x for x in empty_cells if
                                    x[0] in range(first_row, first_row + game_state.board.m) and x[1] in range(
                                        first_column, first_column + game_state.board.n)][0]
                # Place move that is immediately available for point scoring
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], missing_value)
                # Evaluate that move to check how many points it awards
                move_score = self.evaluate_move_score_increase(
                    Move(empty_cell_index[0], empty_cell_index[1], missing_value), game_state, False)
                potential_block_move_points_lost = move_score
                # Remove move from board to return to original game_state
                game_state.board.put(
                    empty_cell_index[0], empty_cell_index[1], SudokuBoard.empty)
            else:
                potential_block_move_points_lost = 0
            score = score - max(potential_row_move_points_lost, potential_col_move_points_lost,
                                potential_block_move_points_lost)

        return score

    def is_possible(self, row_index, column_index, proposed_value, game_state: GameState):
        """
        Determines whether a proposed game move is possible by examining whether the target cell is empty
        and the proposed move in non-taboo.
        @param row_index: The empty cell's row index
        @param col_index: The empty cell's column index
        @param proposed_value: The proposed value to be placed in the specified cell
        @param game_state: The GameState object that describes the game in progress
        @return: True if the proposed move is possible, False otherwise.
        """
        return game_state.board.get(row_index, column_index) == SudokuBoard.empty and not \
            TabooMove(row_index, column_index,
                      proposed_value) in game_state.taboo_moves

    def legal_moves_after_pruning(self, game_state: GameState, empty_cells):
        """
        Filters the provided legal moves using the defined pruning rules.
        @param game_state: The GameState object that describes the game in progress
        @param empty_cells: A list of integer tuples (i, j) representing the coordinates of empty cells
        @return: A list of Move objects representing the result of the legal move filtering process.
        """
        # Prune any cell that we have no information about (the block, row and column containing it are empty). The
        # reasoning behind this pruning is that it is a bit naive to fill in cells for which we have no information
        # and most probably there will be better moves available. This technique significantly reduces the minimax
        # tree size and offers performance advantages.
        # contains all empty cells except the ones for which we have no information
        known_no_reward_cells = []
        if not game_state.board.empty:
            for (row_index, col_index) in empty_cells:
                if not (len(self.get_filled_row_values(row_index, game_state)) == 0 and
                        len(self.get_filled_column_values(col_index, game_state)) == 0 and
                        len(self.get_filled_block_values(row_index, col_index, game_state)) == 0):
                    known_no_reward_cells.append((row_index, col_index))
        else:
            # In the case of an empty board, we assign empty_cells to known_no_reward_cells to avoid pruning all cells
            known_no_reward_cells = empty_cells

        # Filter out illegal moves AND taboo moves from the known_no_reward_cells.
        # The resulting list contains all moves which are both possible and LEGAL
        legal_moves = []
        for coords in known_no_reward_cells:
            for value in self.range_N_plus_1:
                if self.is_possible(coords[0], coords[1], value, game_state) and value not in self.get_illegal_moves(
                        coords[0], coords[1], game_state):
                    legal_moves.append(Move(coords[0], coords[1], value))
        return legal_moves

    def get_empty_cells(self, game_state: GameState):
        """
        Returns the empty cells of the sudoku board at a specified game game_state
        @param game_state: The GameState object that describes the current game_state of the game in progress
        @return: A list of integer tuples (i, j) representing the coordinates of the empty cells
        present in the Sudoku board at its current game game_state
        """
        # Compute empty cells coordinates
        # These are the cells that the agent can probably fill
        empty_cells = [(i, j) for i in self.range_N for j in self.range_N if
                       game_state.board.get(i, j) == SudokuBoard.empty]
        return empty_cells

    def find_optimal_move(self, game_state: GameState, max_depth):
        """
        Used as a helper function that triggers Minimax's recursive call
        @param game_state: The GameState object that describes the current game_state of the game in progress
        @param max_depth: The maximum depth to be reached by Minimax's tree
        @return: A Move object representing the best game move determined through Minimax's recursion
        """
        # Initialize max_score with the lowest possible supported value
        max_score = -math.inf
        # Find all empty cells
        empty_cells = self.get_empty_cells(game_state)

        if len(empty_cells) == 0:
            # Game end, all cells are filled, practically reached a leaf node
            return Move(-1, -1, -1)

        # Initialize best_move to an invalid move
        best_move = Move(-1, -1, -1)
        # Find all possible legal moves for the current game_state
        legal_moves = self.legal_moves_after_pruning(game_state, empty_cells)

        for legal_move in legal_moves:
            # Calculate the amount by which the score of the maximizing player will be increased if it plays legal_move
            score_increase = self.evaluate_move_score_increase(
                legal_move, game_state)
            # Make the move
            game_state.board.put(legal_move.i, legal_move.j, legal_move.value)

            # Increase the score of the player at the current game_state.
            # The score of the maximizing player is saved at game_state.scores[0] and
            # the score of minimizing player is saved at game_state.scores[1]
            if game_state.scores:
                if game_state.scores[0]:
                    game_state.scores[0] += score_increase
                else:
                    game_state.scores[0] = score_increase
            else:
                game_state.scores = [0, score_increase]
            cur_max_score = self.minimax(
                game_state, max_depth, 0, -math.inf, math.inf, False)

            # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
            game_state.board.put(legal_move.i, legal_move.j, SudokuBoard.empty)

            # Undo the score increase to continue by checking other possible moves (recursion unrolling)
            game_state.scores[0] -= score_increase

            if cur_max_score > max_score:
                best_move = Move(legal_move.i, legal_move.j, legal_move.value)
                max_score = cur_max_score
        return best_move

    def get_greedy_move(self, game_state: GameState, legal_moves):
        """
        Returns the move that awards the most points at the current game_state. 
        If there are no moves awarding points, the first move of the list is returned arbitrarily.
        @param game_state: The GameState object that describes the current game_state of the game in progress
        @param legal_moves: All legal moves that can be performed given the current game_state of the game
        @return: A Move object representing the move awarding the mpst points
        """
        max_move = legal_moves[0]
        max_score = -1
        for move in legal_moves:
            max_eval = self.evaluate_move_score_increase(move, game_state)
            if max_eval > max_score:
                max_move = move
                max_score = max_eval
        return max_move

    def minimax(self, game_state: GameState, max_depth: int, depth: int, alpha: float, beta: float,
                is_maximizing_player: bool):
        """
        Implementation of the Minimax algorithm that includes alpha-beta pruning
        @param game_state: The GameState object that describes the current game_state of the game in progress
        @param max_depth: The maximum depth to be reached by Minimax's tree
        @param depth: The current depth reached by Minimax's tree
        @param alpha: The alpha value (used for alpha-beta pruning)
        @param beta: The beta value (used for alpha-beta pruning)
        @param is_maximizing_player: A boolean flag indicating whether it is the maximizing player's turn to play
        @return: The maximum maximizer-minimizer score difference achieved by the Minimax Algorithm
        """
        if depth >= max_depth:
            # Max depth reached, returning the score of the node
            return game_state.scores[0] - game_state.scores[1]

        empty_cells = self.get_empty_cells(game_state)
        # Find any legal moves we can make at the current game_state
        legal_moves = self.legal_moves_after_pruning(game_state, empty_cells)

        if len(legal_moves) == 0:
            # No legal moves left, practically a leaf node was reached. The evaluation function of a node is the
            # difference between the score of the maximizer at this game state and the score of the minimizer at the
            # same game state. This is the quantity that the minimax tries to maximize for the maximizing player and
            # minimize for the opponent.
            return game_state.scores[0] - game_state.scores[1]

        if is_maximizing_player:
            # Maximizer's move
            # Initialize max_score with the lowest possible supported value
            max_score = -math.inf

            for legal_move in legal_moves:
                # Calculate the amount by which the score of the maximizing player will increase if it plays
                # legal_move
                maximizer_score_increase = self.evaluate_move_score_increase(
                    legal_move, game_state)

                # Play the move (add the move to the sudoku board)
                game_state.board.put(
                    legal_move.i, legal_move.j, legal_move.value)

                # In the "scores" property of the GameState object we can find the scores of the maximizing and
                # minimizing players. Since our logic is based on the difference of scores between the maximizing
                # player and minimizing player after a move is played, we need to temporarily reflect the move's
                # result on the player score before continuing the search
                if game_state.scores:
                    if game_state.scores[0]:
                        game_state.scores[0] += maximizer_score_increase
                    else:
                        game_state.scores[0] = maximizer_score_increase
                else:
                    game_state.scores = [maximizer_score_increase, 0]

                # Call minimax for the minimizing player
                max_score = max(max_score, self.minimax(
                    game_state, max_depth, depth + 1, alpha, beta, False))

                # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
                game_state.board.put(
                    legal_move.i, legal_move.j, SudokuBoard.empty)

                # Undo the score increase to continue by checking other possible moves (recursion unrolling)
                game_state.scores[0] -= maximizer_score_increase

                # Implementation of the alpha-beta pruning technique as demonstrated on
                # https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break

            return max_score
        else:
            # minimizer's move
            # Initialize min_score with the highest possible supported value
            min_score = math.inf

            for legal_move in legal_moves:
                # Calculate the amount by which the score of the minimizing player will increase if it plays
                # legal_move
                minimizer_score_increase = self.evaluate_move_score_increase(
                    legal_move, game_state)

                # Play the move (add the move to the sudoku board)
                game_state.board.put(
                    legal_move.i, legal_move.j, legal_move.value)

                # Increase score for the maximizer in the current game_state
                if game_state.scores:
                    if game_state.scores[1]:
                        game_state.scores[1] += minimizer_score_increase
                    else:
                        game_state.scores[1] = minimizer_score_increase
                else:
                    game_state.scores = [0, minimizer_score_increase]

                # Call minimax for the maximizing player
                min_score = min(min_score, self.minimax(
                    game_state, max_depth, depth + 1, alpha, beta, True))

                # Clear legal_move from the board to continue by checking other possible moves (recursion unrolling)
                game_state.board.put(
                    legal_move.i, legal_move.j, SudokuBoard.empty)

                # Undo the score increase to continue by checking other possible moves (recursion unrolling)
                game_state.scores[1] -= minimizer_score_increase

                beta = min(beta, min_score)
                if beta <= alpha:
                    break

            return min_score

    def compute_best_move(self, game_state: GameState) -> None:
        # Filter out illegal moves AND taboo moves
        self.N = game_state.board.N
        self.range_N = range(game_state.board.N)
        self.range_N_plus_1 = range(1, game_state.board.N + 1)
        legal_moves = []
        for i in self.range_N:
            for j in self.range_N:
                for value in self.range_N_plus_1:
                    if self.is_possible(i, j, value, game_state) and value not in self.get_illegal_moves(i, j,
                                                                                                         game_state):
                        legal_moves.append(Move(i, j, value))

        # Propose a valid move arbitrarily at first (random choice from legal moves), to make sure at least "some" move
        # is proposed by our agent in the given time limit
        # start = time.time()
        random_move = random.choice(legal_moves)
        # end = time.time()

       
        # diff = end - start
        # filled_cells = 36 - len(self.get_empty_cells(game_state))
        self.propose_move(random_move)
        # with open("greedy_time_0.5.txt", 'a') as f:
        #     f.write(str(diff) + "      cells filled: " + str(filled_cells) + '\n')
        # f.close()
        # Proceed to propose a "greedy" move. This is slower than proposing a random move, but faster than proposing
        # a minimax move. As the game progresses, greedy moves take less time to be calculated because less cells are
        # empty, thus leaving more time to minimax
        move = self.get_greedy_move(game_state, legal_moves)
        self.propose_move(move)

        # Initial Minimax search depth
        max_depth = 0
        while True:
            # Iteratively increase Minimax's tree depth to discover more optimal moves.
            # On each iteration the move proposed should be slightly better than the move of the previous iteration.
            max_depth += 1
            # Initial call to the recursive minimax function
            best_move = self.find_optimal_move(game_state, max_depth)
            # Fail-safe mechanism to ensure we will never propose an invalid move
            if best_move != Move(-1, -1, -1):
                if self.verbose:
                    # Print statements for debug purposes
                    print("--------------")
                    print("Random move proposed: " + str(best_move))
                    print("Score for selected legal_move: " + str(
                        self.evaluate_move_score_increase(best_move, game_state)))
                    print("Illegal moves for selected cell: " + str(
                        self.get_illegal_moves(best_move.i, best_move.j, game_state)))
                    print("Block filled values for selected cell: " + str(
                        self.get_filled_block_values(best_move.i, best_move.j, game_state)))
                    print("Row filled values for selected cell: " + str(
                        self.get_filled_row_values(best_move.i, game_state)))
                    print("Column filled values for selected cell: " + str(
                        self.get_filled_column_values(best_move.j, game_state)))
                    print("--------------")
                self.propose_move(best_move)
