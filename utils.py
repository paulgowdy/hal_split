import numpy as np
import config as cfg


def process_obs(obs):
    '''
    obs is a dictionary that looks like this:

    {'player': 0, # I believe this is the 'current' player - i.e. the player were controlling
    'step': 0,
    'halite': [25, 0, 100, 0, 25, 0, 0, 25, 0, 0, 0, 100, 0, 100, 0, 0, 0, 25, 0, 0, 25, 0, 100, 0, 25],
    'players': [[5000, {}, {'0-1': [12, 0]}]]}
    '''
    player_ships = obs['players'][0][2]
    player_yards = obs['players'][0][1]

    halite_layer = np.array(obs['halite']).reshape((cfg.BOARD_SIZE, cfg.BOARD_SIZE))
    ship_layer = [0] * cfg.BOARD_SIZE ** 2
    yard_layer = [0] * cfg.BOARD_SIZE ** 2

    ship_select_layers = np.zeros((cfg.BOARD_SIZE ** 2, cfg.MAX_NB_SHIPS))

    for i, ship in enumerate(list(player_ships.values())):
        ship_layer[ship[0]] = (ship[1] + 1)/100.
        # Positive values for my ships, negative values for enemy ships
        ship_select_layers[ship[0], i] = 1.

    for yard_loc in list(player_yards.values()):
        yard_layer[yard_loc] = 1.

    ship_layer = np.array(ship_layer).reshape((cfg.BOARD_SIZE, cfg.BOARD_SIZE))
    yard_layer = np.array(yard_layer).reshape((cfg.BOARD_SIZE, cfg.BOARD_SIZE))

    ship_select_layers = ship_select_layers.reshape((cfg.BOARD_SIZE, cfg.BOARD_SIZE, cfg.MAX_NB_SHIPS))

    board_obs = np.stack([halite_layer, ship_layer, yard_layer], axis=-1)

    #print(board_obs.shape)
    #print(halite_layer)
    #print(ship_layer)
    #print(yard_layer)

    #print(ship_select_layers.shape)
    #print(ship_select_layers[:,:,0])
    #print(ship_select_layers[:,:,1])
    #print(ship_select_layers[:,:,2])

    scalar_obs = np.array([obs['step']/float(cfg.STEPS_PER_EP), (obs['players'][0][0] - 4000.)/1000.])

    return board_obs, scalar_obs, ship_select_layers
