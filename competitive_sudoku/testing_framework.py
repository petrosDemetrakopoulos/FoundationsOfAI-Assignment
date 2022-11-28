import os
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def run_game(player_1_name, player_2_name):
    command = "python simulate_game.py --first {} --second {} --time 1".format(player_1_name, player_2_name)
    r = os.popen(command) #Execute command
    info = r.readlines()  #read command output

    output_lines = []
    for line in info:  #handle output line by line
        line = line.strip('\r\n')
        output_lines.append(line)
        # print(line)

    last_output_line = output_lines[len(output_lines) - 1]
    # print("The last line printed is: ", last_output_line)

    if last_output_line == "Player 1 wins the game.":
        return player_1_name
    elif last_output_line == "Player 2 wins the game.":
        return player_2_name
    elif last_output_line == "The game ends in a draw.":
        return None
    else:
        raise Exception("Unexpected last_output_line value!")


if __name__ == '__main__':
    num_of_runs = 100
    num_of_threads = 100
    print("> Active Threads: ", num_of_threads)
    print("\n")

    print("*** TEST SESSION BEGAN ***")

    agent_1 = Agent("team09_A1", True)
    agent_2 = Agent("random_player", False)

    with ThreadPoolExecutor(num_of_threads) as executor:
        futures = []

        player_1_id = agent_1.get_name()
        player_2_id = agent_2.get_name()
        for i in range(num_of_runs):
            futures.append(executor.submit(run_game, player_1_id, player_2_id))
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

    print("*** TEST RESULTS ***")
    print(" \n     > Overview <")
    print("----------------------")
    print("> #Game Runs: ", num_of_runs)
    print("> #(Our) Custom Agent Wins: ", agent_1.get_wins() if agent_1.is_custom() else agent_2.get_wins())
    print("> #Opponent Agent Wins: ", agent_2.get_wins() if not agent_2.is_custom() else agent_1.get_wins())
    print("> #Draws: ", num_of_runs - (agent_1.get_wins() + agent_2.get_wins()))
    print("----------------------")

    print(" \n   > Win Rates (%) <")
    print("----------------------")
    print("Our Agent: {0:.2f}%".format(agent_1.get_win_rate()))
    print("Opponent Agent: {0:.2f}%".format(agent_2.get_win_rate()))
    print("----------------------")
