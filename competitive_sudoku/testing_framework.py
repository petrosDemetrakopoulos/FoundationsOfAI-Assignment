import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        log_file_name = "logfile.log"
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        self.log = open(log_file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()


class Agent:
    def __init__(self, name, custom_agent):
        self.name = name
        self.custom_agent = custom_agent

        self.wins = 0
        self.losses = 0
        self.draws = 0

    def increment_wins(self):
        self.wins += 1

    def increment_losses(self):
        self.losses += 1

    def increment_draws(self):
        self.draws += 1

    def is_custom(self):
        return self.custom_agent

    def get_wins(self):
        return self.wins

    def get_win_rate(self):
        return self.wins / (self.wins + self.losses + self.draws) * 100

    def get_name(self):
        return self.name


def run_game(player_1_name, player_2_name, time, board):
    command = "python simulate_game.py --first {} --second {} --time {} --board {}".format(player_1_name, player_2_name, time, board)
    r = os.popen(command) #Execute command
    info = r.readlines()  #read command output

    output_lines = []
    for line in info:  #handle output line by line
        line = line.strip('\r\n')
        output_lines.append(line)
        # print(line)

    last_output_line = output_lines[len(output_lines) - 1]
    # print("The last line printed is: ", last_output_line)

    if "Player 1 wins the game." in last_output_line:
        return player_1_name
    elif "Player 2 wins the game." in last_output_line:
        return player_2_name
    elif "The game ends in a draw." in last_output_line:
        return None
    else:
        print("DEBUG - last_output_line: ", last_output_line)
        raise Exception("Unexpected last_output_line value!")


def run_test_scenario(time_option, scenario_name, agent_1_name, agent_2_name, num_of_runs, num_of_threads):
    print('*** TEST SESSION (Time: {}, Scenario: "{}", Agents: "{}", "{}") IN PROGRESS ***'.format(time_option, scenario_name, agent_1_name, agent_2_name))

    agent_1 = Agent(agent_1_name, True)
    agent_2 = Agent(agent_2_name, False)

    with ThreadPoolExecutor(num_of_threads) as executor:
        futures = []

        player_1_id = agent_1.get_name()
        player_2_id = agent_2.get_name()
        for i in range(num_of_runs):
            futures.append(executor.submit(run_game, player_1_id, player_2_id, time_option, test_file))
            player_1_id, player_2_id = player_2_id, player_1_id

    # process each result as it is available
    finished_count = 0
    for future in as_completed(futures):
        winner_name = future.result()
        finished_count += 1
        print("> Finished running game {} out of {}".format(finished_count, num_of_runs))

        if winner_name is None:
            agent_1.increment_draws()
            agent_2.increment_draws()
        elif winner_name == agent_1.get_name():
            agent_1.increment_wins()
            agent_2.increment_losses()
        elif winner_name == agent_2.get_name():
            agent_2.increment_wins()
            agent_1.increment_losses()

    print('*** RESULTS (Time: {}, Scenario: "{}") ***'.format(time_option, scenario_name))
    print(" \n     > Overview <")
    print("----------------------")
    print("> #Game Runs: ", num_of_runs)
    print("> #{} Wins: {}".format(agent_1.get_name(), agent_1.get_wins()))
    print("> #{} Wins: {}".format(agent_2.get_name(), agent_2.get_wins()))
    print("> #Draws: ", num_of_runs - (agent_1.get_wins() + agent_2.get_wins()))
    print("----------------------")

    print(" \n   > Win Rates (%) <")
    print("----------------------")
    print("{}: {:.2f}%".format(agent_1.get_name(), agent_1.get_win_rate()))
    print("{}: {:.2f}%".format(agent_2.get_name(), agent_2.get_win_rate()))
    print("----------------------")


if __name__ == '__main__':
    num_of_runs = 10
    num_of_threads = 10
    test_files_root_path = "./boards"
    test_files_names = ["easy-2x2.txt",
                      "easy-3x3.txt",
                      "empty-2x2.txt",
                      "empty-2x3.txt",
                      "empty-3x3.txt",
                      "empty-3x4.txt",
                      "empty-4x4.txt",
                      "hard-3x3.txt",
                      "random-2x3.txt",
                      "random-3x3.txt",
                      "random-3x4.txt",
                      "random-4x4.txt"]
    test_files_names = ['empty-4x4.txt']

    test_files_paths = [test_files_root_path + "/" + file_name for file_name in test_files_names]

    agent_1_name = "team09_A2"
    agent_2_name = "greedy_player"
    time_options = [0.1, 0.5, 1]

    print(" \n     > Script Settings <")
    print("--------------------------------------------")
    print("> #Game Runs/Test Scenario: ", num_of_runs)
    print("> #Active Threads: ", num_of_threads)
    print("--------------------------------------------")

    print(" \n     > Loaded Test Scenarios <")
    print("--------------------------------------------")
    print("> #Loaded Test Scenarios: {}\n".format(len(test_files_paths)))
    for test_file in test_files_paths:
        print(test_file)
    print("--------------------------------------------")
    print("\n")

    # Run the game for every time option
    for time_option in time_options:
        # Run the game for every loaded test scenario
        for test_file in test_files_paths:
            print("=======================================================================================================")
            run_test_scenario(time_option, test_file, agent_1_name, agent_2_name, num_of_runs, num_of_threads)
            print("\n=======================================================================================================")

