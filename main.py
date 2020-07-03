"""Run Halite simulations for training purposes"""
from kaggle_environments import make
from kaggle_environments.envs.halite import helpers

import torch
import torch.nn as nn
import torch.nn.functional as F

# Idea: Track & characterise opponent strategy as some sort of vector
# Idea: The final score is what's important. How to have a good reward function that prioritises shorter term rewards
#   towards the end of the game?
# Idea: Should have multiple populations of the AI, so that it can train against different strategies?
#   Or have some inputs to the AI be a random vector, determined at the beginning of the game? (A personality vector?)

class AI():

    def __init__(self):
        self.personality = (0, 0, 0, 0, 0)  # TODO: Random vector
        self.opponent_personality_estimate = (0, 0, 0, 0, 0)  # Fit to the opponents previous move!?


    def run(self, observation, configuration):
        """
        Queues next actions for each ship & shipyard and returns them.
        :param observation:
        :param configuration:
        :return: The set of player actions which can be passed to environment.run
        """
        board = helpers.Board(raw_observation=observation, raw_configuration=configuration)
        current_player = board.current_player
        for ship in current_player.ships:
            ship.next_action = self.ship_ai(board, ship.id)
        for shipyard in current_player.shipyards:
            shipyard.next_action = self.shipyard_ai(board, ship.id)

        return current_player.next_actions


def run_episode(AI, board_size: int = 21, max_turns: int = 400, starting_halite: int = 5000, agent_count: int = 4):
    """
    Runs a complete episode with a given AI and game settings.
    :param AI: The AI function to call
    :param board_size: The size of the board to use
    :param max_turns: The turn at which the scores will be totaled & the winner decided
    :param starting_halite: The amount of Halite that each agent starts with
    :param agent_count: How many agents should be simulated
    :return: The final scores of each player
    """
    environment = make("halite", configuration={"size": board_size, "startingHalite": starting_halite})
    environment.reset(num_agents=agent_count)
    state = environment.state[0]
    board = helpers.Board(raw_observation=state.observation, raw_configuration=environment.configuration)

    environment.run(["random", "random"])
    out = environment.render(mode="html", width=500, height=450)

    f = open("halite.html", "w")
    f.write(out)
    f.close()


if __name__ == "__main__":
    run_episode(None)