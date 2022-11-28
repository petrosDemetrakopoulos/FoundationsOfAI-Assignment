import os
from concurrent.futures import ThreadPoolExecutor, as_completed


class Player:
    def __init__(self, name, is_custom_agent):
        self.name = name
        self.is_custom_agentg = is_custom_agent


def run_game(first_player, second_player):
    command = "python simulate_game.py --first {} --second {} --time 1".format(first_player, second_player)
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
        return 1
    elif last_output_line == "Player 2 wins the game.":
        return 2
    elif last_output_line == "The game ends in a draw.":
        return 0


if __name__ == '__main__':
    num_of_runs = 100
    num_of_threads = 100
    print("> Active Threads: ", num_of_threads)
    print("\n")

    print("*** TEST SESSION BEGAN ***")
    agent_wins = 0
    opponent_wins = 0
    draws = 0
    P1_is_agent = True
    player_1 = "team09_A1"
    player_2 = "random_player"
    with ThreadPoolExecutor(num_of_threads) as executor:
        futures = []

        for i in range(num_of_runs):
            futures.append(executor.submit(run_game, player_1, player_2))
            P1_is_agent = not P1_is_agent
            player_1, player_2 = player_2, player_1

    # process each result as it is available
    finished_count = 0
    for future in as_completed(futures):
        result_int = future.result()
        finished_count += 1
        print("> Finished running game {} out of {}".format(finished_count, num_of_runs))

        if result_int == 0:
            draws += 1
        elif result_int == 1:
            if P1_is_agent:
                agent_wins += 1
            else:
                opponent_wins += 1
        elif result_int == 2:
            if not P1_is_agent:
                agent_wins += 1
            else:
                opponent_wins += 1

    print("*** TEST RESULTS ***")

    print(" \n     > Overview <")
    print("----------------------")
    print("> #Game runs: ", num_of_runs)
    print("> #Agent wins: ", agent_wins)
    print("> #Opponent wins: ", opponent_wins)
    print("> #Draws: ", draws)
    print("----------------------")

    print(" \n   > Win Rates (%) <")
    print("----------------------")
    print("Agent: {0:.2f}%".format(agent_wins / num_of_runs * 100))
    print("Opponent: {0:.2f}%".format(opponent_wins / num_of_runs * 100))
    print("----------------------")
