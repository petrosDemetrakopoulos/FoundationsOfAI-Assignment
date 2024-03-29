#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)
import copy
import random
import numpy as np

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


# Implementation of MCTS is based on https://ai-boson.github.io/mcts/
class TreeNode:
    # Class representing the nodes of Monte Carlo Tree Search
    def __init__(self, game_state: GameState, parent_node, parent_move, candidate_moves, num_empty_cells, is_our_turn=True):
        self.game_state = game_state                # The current game state object
        self.parent_node = parent_node              # The TreeNode object representing this node's parent node
        self.parent_move = parent_move              # The move selected by the parent node of this node
        self.children_nodes = []                    # The children nodes of this node
        self.candidate_moves = candidate_moves      # All available moves provided as potential choices to this node
        self.num_empty_cells = num_empty_cells      # The number of empty cells in the current game state
        self.is_our_turn = is_our_turn              # Flag that indicates whether it is our agent's turn to play

        self.n_value = 0                            # Number of times the node has been visited
        self.win_count = [0, 0]                     # The win counts of both players
                                                    # (index 0: Our agent, index 1: Opponent)
        self.unevaluated_moves = candidate_moves    # A collection of moves that have not been evaluated
                                                    # in terms of potential score yet
        return

    def get_q_value(self):
        """
        Getter method used to return the accumulated "q" (score) value of a node
        @return: an integer representing the accumulated "q" value
        """
        p1_wins = self.win_count[0]
        p1_loses = self.win_count[1]

        return p1_wins - p1_loses

    def get_n_value(self):
        """
        Getter method used to return the accumulated "n" (number of visits) value of a node
        @return: an integer representing the "n" value
        """
        return self.n_value

    def get_parent_move(self):
        """
        Getter method used to get the stored parent move of the current node.
        @return: a Move object representing this node's parent move.
        """
        return self.parent_move

    def expand_tree(self):
        """
        Method  used to perform the "tree expansion" step of Monte Carlo Tree Search
        @return: the child node of the current node that was created as a result of the tree expansion
        """
        # Pick the next move from the unevaluated moves list
        new_move = self.unevaluated_moves.pop()  # Choose a random unevaluated move to evaluate
        # Calculate the new move's score
        new_move_score = evaluate_move_score_increase(new_move, self.game_state)

        if self.is_our_turn:
            self.game_state.scores[0] += new_move_score
        else:
            self.game_state.scores[1] += new_move_score

        new_game_state = copy.deepcopy(self.game_state)  # Create a copy of the current game state
        new_game_state.board.put(new_move.i, new_move.j, new_move.value)  # Make the new move on the new game state

        updated_empty_cells = get_empty_cells(new_game_state)

        # Find any legal moves we can make at the current game_state
        updated_candidate_moves = legal_moves_after_pruning(new_game_state, updated_empty_cells)

        child_node = TreeNode(new_game_state, self, new_move, updated_candidate_moves, self.num_empty_cells - 1,
                              not self.is_our_turn)

        self.children_nodes.append(child_node)

        return child_node

    def select_rollout_move(self, possible_moves, game_state):
        """
        Method used to select a rollout move from a list of possible moves and for a given game state, every time it
        is called. This method is called during every iteration of the simulation "step" of Monte Carlo Search Tree
        @param possible_moves: a list of "Move" objects representing possible moves
        @param game_state: a GameState object representing the current state of the Competitive Sudoku game
        @return: a "Move" object representing the selected rollout move
        """
        # Sort moves based on their score increase (moves leading to higher score first)
        scores = [evaluate_move_score_increase(move, game_state) for move in possible_moves]
        moves = [x[1] for x in sorted(zip(scores, possible_moves), key=lambda tup: tup[0])]
        moves.reverse()

        # Pick 1 move out of the best k moves
        k = 5
        if len(moves) >= k:
            moves = moves[:k]
    
        return moves[np.random.randint(len(moves))]

    def is_terminal_node(self):
        """
        Helper method used to determine if the current node is a terminal (leaf) node
        @return: a boolean value (True if the current node is a leaf, False otherwise)
        """
        return len(get_empty_cells(self.game_state)) == 0

    def rollout(self):
        """
        Method used to perform the roll-out (simulation) step of Monte Carlo Search Tree
        @return: the accumulated scores of both players from the executed roll-out (simulation)
        """
        # The simulation (rollout) starts one level below the current node (next move), therefore
        # we change the player flag to indicate that it is the next player's turn
        is_our_turn = not self.is_our_turn
        rollout_game_state = copy.deepcopy(self.game_state)
        available_moves = self.candidate_moves

        while available_moves:
            # Select a random move from the available moves
            selected_move = self.select_rollout_move(available_moves, game_state=rollout_game_state)

            # Calculate the selected move's score
            selected_move_score = evaluate_move_score_increase(selected_move, rollout_game_state)

            # "Play" the move on the board
            rollout_game_state.board.put(selected_move.i, selected_move.j, selected_move.value)

            # Update the saved score for the player that is currently playing
            if is_our_turn:
                rollout_game_state.scores[0] += selected_move_score
            else:
                rollout_game_state.scores[1] += selected_move_score

            # Change the player order flag to the next player
            is_our_turn = not is_our_turn

            # Update the available moves
            empty_cells = get_empty_cells(rollout_game_state)
            available_moves = legal_moves_after_pruning(rollout_game_state, empty_cells)

        return rollout_game_state.scores

    def backpropagate(self, result):
        """
        Method used to perform the back-propagation step of Monte Carlo Search Tree
        @param result: a list containing the accumulated scores (game points) of each player after
        the execution of the roll-out (simulation) step.
        """
        self.n_value += 1.

        # Check which player has the most turn wins based on the provided result
        if result[0] > result[1]:
            # Player1 has the most wins
            self.win_count[0] += 1
        elif result[1] > result[0]:
            self.win_count[1] += 1

        # If a parent node exists (current node is NOT root), backpropagate the result
        if self.parent_node:
            self.parent_node.backpropagate(result)

    def is_fully_expanded(self):
        """
        Helper method used to check if the current node is fully expanded (has no unevaluated moves left)
        @return: a boolean value (True if the node has no unevaluated moves left, False otherwise)
        """
        return len(self.unevaluated_moves) == 0

    def get_best_child(self, c_param=2):
        """
        Method implementing the UCT formula to select and return the best child of the current node
        @param c_param: a hyper-parameter used to tweak the formula's sensitivity/performance
        @return: the best child node of the current node selected using the UCT formula
        """
        choices_weights = [
            (c.get_q_value() / c.get_n_value()) + c_param * np.sqrt((2 * np.log(self.get_n_value()) / c.get_n_value()))
            for c in self.children_nodes]
        return self.children_nodes[np.argmax(choices_weights)]

    def select_rollout_node(self):
        """
        Method used to select the roll-out node i.e., the node where the simulation step will commence from.
        @return: the selected roll-out node.
        """
        current_node = self

        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand_tree()
            else:
                current_node = current_node.get_best_child(c_param=2)

        return current_node


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    verbose = False  # a flag to print useful debug logs after each turn

    def __init__(self):
        super().__init__()
        self.move_skipped = False

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
            max_eval = evaluate_move_score_increase(move, game_state)
            if max_eval > max_score:
                max_move = move
                max_score = max_eval
        return max_move

    def monte_carlo_tree_search(self, game_state: GameState, candidate_moves, num_simulations):
        """
        Helper method used to trigger the Monte Carlo Search Tree algorithm
        @param game_state: the current state of the Competitive Sudoku game
        @param candidate_moves: a list of "Move" objects containing all possible moves that should be processed
        @param num_simulations: The number of simulation we want the Monte Carlo Search Tree algorithm to perform
        """
        num_empty_cells = len(get_empty_cells(game_state))
        root_node = TreeNode(game_state, None, None, candidate_moves, num_empty_cells)

        for i in range(num_simulations):
            v = root_node.select_rollout_node()
            result = v.rollout()
            v.backpropagate(result)

            best_move = root_node.get_best_child(c_param=2).get_parent_move()
            self.propose_move(best_move)

    def get_skip_move(self, legal_moves: list, game_state: GameState):
        """
        Get a move that would make the Sudoku unsolvable, to use it to intentionally skip the agent's move
        @param game_state: The GameState object that describes the current state of the game in progress
        @param legal_moves: List of all legal moves that are available to the agent at the current state
        @return: A Move object representing a move that makes the sudoku unsolvable. If such moves does not exists, return None
        """

        # iterate rows to find potential move than can force the agent to "skip" the move
        for i in range(game_state.board.N):
            available_moves_in_row = [move for move in legal_moves if move.i == i]
            available_cells_in_row = list(set([move.j for move in available_moves_in_row]))
            moves_per_cell = {col_index : [] for col_index in available_cells_in_row}
            for move in available_moves_in_row:
                moves_per_cell[move.j].append(move.value)
            non_ambiguous_value = None
            for col_index, moves_list in moves_per_cell.items():
                if len(moves_list) == 1:
                    non_ambiguous_value = moves_list[0]

            if non_ambiguous_value is not None:
                # If a cell in the row can contain only a single value
                # but another cell from the row can also have it
                # then it means that putting the value in the latter one will result in an unsolvable sudoku
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
            non_ambiguous_value = None
            for row_index, moves_list in moves_per_cell.items():
                if len(moves_list) == 1:
                    non_ambiguous_value = moves_list[0]

            if non_ambiguous_value is not None:
                # If a cell in the row can contain only a single value
                # but another cell from the row can also have it
                # then it means that putting the value in the latter one will result in an unsolvable sudoku
                # we want to propose that move in order to "skip" the turn
                for row_index, moves_list in moves_per_cell.items():
                    if len(moves_list) > 1 and non_ambiguous_value in moves_list:
                        return Move(row_index, j, non_ambiguous_value)
        
        return None

    def compute_best_move(self, game_state: GameState) -> None:
        # Initialize GameState.scores with 0 for both players
        game_state.scores = [0, 0]

        # Filter out illegal moves AND taboo moves
        range_N = range(game_state.board.N)
        range_N_plus_1 = range(1, game_state.board.N + 1)
        legal_moves = []
        for i in range_N:
            for j in range_N:
                for value in range_N_plus_1:
                    if is_possible(i, j, value, game_state) and value not in get_illegal_moves(i, j, game_state):
                        legal_moves.append(Move(i, j, value))

        # Propose a valid move arbitrarily at first (random choice from legal moves), to make sure at least "some" move
        # is proposed by our agent in the given time limit
        random_move = random.choice(legal_moves)
        self.propose_move(random_move)

        # Propose a greedy move (the highest-scoring move available at the current game state)
        move = self.get_greedy_move(game_state, legal_moves)
        self.propose_move(move)

        empty_cells_count = len(get_empty_cells(game_state))

        # Check whether skipping a move is to the best of our interest here using our "skip" move logic
        if empty_cells_count % 2 == 0:
            skip_move = self.get_skip_move(legal_moves, game_state)
            if skip_move is not None:
                self.propose_move(skip_move)
                self.move_skipped = True
        
        if not self.move_skipped:
            # Monte Carlo Search Tree
            num_simulations = 1000000
            self.monte_carlo_tree_search(game_state, legal_moves, num_simulations)


###### Start of helper functions ######
def get_filled_row_values(row_index: int, game_state: GameState):
    """
    Returns the non-empty values of the row specified by a given row index.
    @param row_index: The row index
    @param game_state: The GameState object that describes the game in progress
    @return: A list containing the integer values of the specified row's non-empty cells
    """
    # returns non-empty values in row with index row_index
    filled_values = []
    for i in range(game_state.board.N):
        cur_cell = game_state.board.get(row_index, i)
        if cur_cell != SudokuBoard.empty:
            filled_values.append(cur_cell)
    return filled_values


def get_filled_column_values(column_index: int, game_state: GameState):
    """
    Returns the non-empty values of the column specified by a given column index.
    @param column_index: The column index
    @param game_state: The GameState object that describes the game in progress
    @return: A list containing the integer values of the specified column's non-empty cells.
    """
    filled_values = []
    for i in range(game_state.board.N):
        cur_cell = game_state.board.get(i, column_index)
        if cur_cell != SudokuBoard.empty:
            filled_values.append(cur_cell)
    return filled_values


def get_filled_block_values(row_index: int, column_index: int, game_state: GameState):
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


def is_possible(row_index, column_index, proposed_value, game_state: GameState):
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


def get_illegal_moves(row_index: int, col_index: int, game_state: GameState):
    """
    Returns a list of numbers that already exist in the specified cell's row, column or block. These numbers
    are illegal values and CANNOT be put on the given empty cell.
    @param row_index: The empty cell's row index
    @param col_index: The empty cell's column index
    @param game_state: The GameState object that describes the game in progress
    @return: A list of integers representing the illegal values of the specified empty cell.
    """
    illegal = get_filled_row_values(row_index, game_state) + get_filled_column_values(col_index, game_state) + get_filled_block_values(row_index, col_index, game_state)
    return set(illegal)  # Easy way to remove duplicates


def legal_moves_after_pruning(game_state: GameState, empty_cells):
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
            if not (len(get_filled_row_values(row_index, game_state)) == 0 and
                    len(get_filled_column_values(col_index, game_state)) == 0 and
                    len(get_filled_block_values(row_index, col_index, game_state)) == 0):
                known_no_reward_cells.append((row_index, col_index))
    else:
        # In the case of an empty board, we assign empty_cells to known_no_reward_cells to avoid pruning all cells
        known_no_reward_cells = empty_cells

    # Filter out illegal moves AND taboo moves from the known_no_reward_cells.
    # The resulting list contains all moves which are both possible and LEGAL
    legal_moves = []
    for coords in known_no_reward_cells:
        for value in range(game_state.board.N + 1):
            if is_possible(coords[0], coords[1], value, game_state) and value not in get_illegal_moves(
                    coords[0], coords[1], game_state):
                legal_moves.append(Move(coords[0], coords[1], value))
    return legal_moves


def get_empty_cells(game_state: GameState):
    """
    Returns the empty cells of the sudoku board at a specified game game_state
    @param game_state: The GameState object that describes the current game_state of the game in progress
    @return: A list of integer tuples (i, j) representing the coordinates of the empty cells
    present in the Sudoku board at its current game game_state
    """
    # Compute empty cells coordinates
    # These are the cells that the agent can probably fill
    board_size = game_state.board.N
    empty_cells = [(i, j) for i in range(board_size) for j in range(board_size) if
                   game_state.board.get(i, j) == SudokuBoard.empty]
    return empty_cells


def evaluate_move_score_increase(move: Move, game_state: GameState, allow_recusion=True):
    """
    Calculates the score increase achieved after the proposed move is made.
    @param move: A Move object that describes the proposed move
    @param game_state: The GameState object that describes the game in progress
    @return: The calculated score increase achieved by the proposed move
    """
    filled_row = get_filled_row_values(move.i, game_state)
    filled_col = get_filled_row_values(move.j, game_state)
    filled_block = get_filled_block_values(move.i, move.j, game_state)

    full_len = game_state.board.N - 1
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
        empty_cells = get_empty_cells(game_state)
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
            move_score = evaluate_move_score_increase(
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
            move_score = evaluate_move_score_increase(
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
            move_score = evaluate_move_score_increase(
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
