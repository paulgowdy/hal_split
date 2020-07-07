from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import numpy as np

import config as cfg

class HaliteEnvironment:

    def __init__(self, board_size=5, startingHalite=1000):

        self.agent_count = 1
        self.board_size = board_size
        self.max_nb_ships = 1

        self.environment = make("halite", configuration={"size": board_size, "startingHalite": startingHalite})
        self.environment.reset(self.agent_count)

        state = self.environment.state[0]
        self.board = Board(state.observation, self.environment.configuration)

        # Ship actions in order:
        # [Hold, North, East, South, West, Convert]
        self.ship_action_conversion_dict = {
        0:None,
        1:ShipAction.NORTH,
        2:ShipAction.EAST,
        3:ShipAction.SOUTH,
        4:ShipAction.WEST,
        5:ShipAction.CONVERT
        }

    def step(self, ship_actions, yard_actions):
        # Ship actions will be a list of integers
        # The order of the list will match players[0][2].keys()
        # Ints will be from 0 to cfg.NB_SHIP_ACTIONS-1

        initial_halite = self.board.observation['players'][0][0]


        ship_actions_converted = [self.convert_ship_action(x) for x in ship_actions]


        for a, ship_id in zip(ship_actions_converted, iter(self.board.observation['players'][0][2].keys())):
            self.board.ships[ship_id].next_action = a

        for a, yard_id in zip(yard_actions, iter(self.board.observation['players'][0][1].keys())):
            if a == 1:
                self.board.shipyards[yard_id].next_action = ShipyardAction.SPAWN

        self.board = self.board.next()
        next_obs = self.board.observation

        post_move_halite = self.board.observation['players'][0][0]
        reward = (post_move_halite - initial_halite)/100.

        terminal = False
        if self.board.observation['step'] == cfg.STEPS_PER_EP:
            terminal = True

        return next_obs, reward, terminal

    def convert_ship_action(self, action):
        return self.ship_action_conversion_dict[action]

    def reset(self):
        self.environment.reset(self.agent_count)
        state = self.environment.state[0]
        self.board = Board(state.observation, self.environment.configuration)
        return state.observation
