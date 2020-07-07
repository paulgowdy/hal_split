import config as cfg
from halite_environment import HaliteEnvironment
from utils import process_obs
from halite_actor import create_actor_model
from halite_critic import create_critic_model
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)

def take_starting_actions(env):
    # takes a fresh env
    env.step([5], [])
    env.step([], [1])
    env.step([1], [1])
    #env.step([0,2], [1])
    #env.step([0,0,3], [1])
    #env.step([0,0,0,4], [])

    return env.board.observation

# Create environment
env = HaliteEnvironment(cfg.BOARD_SIZE)
# Reset env

#Process obs
actor = create_actor_model()
critic = create_critic_model()


'''
z = actor.predict([np.expand_dims(current_board,0),
                    np.expand_dims(current_scalars,0),
                    np.expand_dims(current_ship_select,0),
                    np.zeros((1,1)),
                    np.random.random((1,4,5))], batch_size=1)

print(z)

s = critic.predict([np.expand_dims(current_board,0), np.expand_dims(current_scalars,0)])
print(s)
'''



def process_action_probs(action_probs):
    actions = []

    for i in range(action_probs.shape[0]):
        if np.sum(action_probs[i,:]) > 0.:
            actions.append(np.random.choice(cfg.NB_SHIP_ACTIONS, p=np.nan_to_num(action_probs[i])))

    return actions

def unroll_rewards(rewards):
    new_reward = list(rewards)

    for j in range(len(new_reward) - 2, -1, -1):
        new_reward[j] += new_reward[j + 1] * cfg.GAMMA

    return new_reward

raw_obs = env.reset()
#print(env.board)
raw_obs = take_starting_actions(env)
#print(env.board)
#print(env.board.observation)
current_board, current_scalars, current_ship_select = process_obs(raw_obs)
reward_collect = []

ep_reward_collect = []
actor_loss_collect = []
critic_loss_collect = []

ep_counter = 0

for b in range(cfg.PPO_BATCHES):
    print('Batch:', b)

    batch = {
    'board_obs': [],
    'scalar_obs': [],
    'ship_select_obs': [],

    'action': [],
    'pred': [],
    'reward': [],
    }

    temp_batch = {
    'board_obs': [],
    'scalar_obs': [],
    'ship_select_obs': [],

    'action': [],
    'pred': [],
    #'reward': [],
    }

    #for step in range(16):
    while len(batch['board_obs']) < 128:
        #print('Step', step)
        ship_orders_probs_flat = actor.predict([np.expand_dims(current_board,0),
                            np.expand_dims(current_scalars,0),
                            np.expand_dims(current_ship_select,0),
                            np.zeros((1,1)),
                            np.random.random((1,cfg.MAX_NB_SHIPS * cfg.NB_SHIP_ACTIONS))])

        #print(ship_orders_probs_flat)
        ship_orders_probs_rs = np.reshape(ship_orders_probs_flat, (cfg.MAX_NB_SHIPS, cfg.NB_SHIP_ACTIONS))
        #print('sop_rs', ship_orders_probs_rs.shape)
        #print(process_action_probs(ship_orders_probs))
        order_list = process_action_probs(ship_orders_probs_rs)
        action_matrix = np.zeros((cfg.MAX_NB_SHIPS, cfg.NB_SHIP_ACTIONS))
        action_matrix[np.arange(len(order_list)),order_list] = 1.
        #print('action matrix', action_matrix.shape)
        #print(order_list)
        action_matrix_flat = np.reshape(action_matrix, (cfg.MAX_NB_SHIPS * cfg.NB_SHIP_ACTIONS,))
        #print('action matrix_rs', action_matrix_rs.shape)

        next_obs, reward, terminal = env.step(order_list, [])

        #print('Env step:', env.board.observation['step'])


        reward_collect.append(reward)

        temp_batch['board_obs'].append(current_board)
        temp_batch['scalar_obs'].append(current_scalars)
        temp_batch['ship_select_obs'].append(current_ship_select)

        temp_batch['action'].append(action_matrix_flat)
        temp_batch['pred'].append(ship_orders_probs_flat)

        # Be sure to collect all the obs before converting again!
        current_board, current_scalars, current_ship_select = process_obs(next_obs)

        if terminal:
            print('Episode ending!', ep_counter)
            #print(reward_collect)
            reward_collect = unroll_rewards(reward_collect)
            #print(reward_collect)
            ep_counter += 1
            ep_reward_collect.append(env.board.observation['players'][0][0]-2500.)

            for j in range(len(temp_batch['board_obs'])):
                batch['board_obs'].append(temp_batch['board_obs'][j])
                batch['scalar_obs'].append(temp_batch['scalar_obs'][j])
                batch['ship_select_obs'].append(temp_batch['ship_select_obs'][j])

                batch['action'].append(temp_batch['action'][j])
                batch['pred'].append(temp_batch['pred'][j])
                batch['reward'].append(reward_collect[j])

            temp_batch = {
            'board_obs': [],
            'scalar_obs': [],
            'ship_select_obs': [],

            'action': [],
            'pred': [],
            #'reward': [],
            }

            raw_obs = env.reset()
            raw_obs = take_starting_actions(env)
            current_board, current_scalars, current_ship_select = process_obs(raw_obs)
            reward_collect = []


    board_obs, scalar_obs, ship_select_obs = np.array(batch['board_obs']), np.array(batch['scalar_obs']), np.array(batch['ship_select_obs']),
    actions, preds, rewards = np.array(batch['action']), np.reshape(np.array(batch['pred']), (len(batch['pred']),cfg.MAX_NB_SHIPS * cfg.NB_SHIP_ACTIONS)),  np.reshape(np.array(batch['reward']), (len(batch['reward']), 1))

    #print(board_obs.shape, scalar_obs.shape, ship_select_obs.shape)
    #print(actions.shape, preds.shape)#, rewards.shape)
    predicted_values = critic.predict([board_obs, scalar_obs])
    #print(predicted_values.shape)
    #print(actions[0])
    #print(preds[0])

    advantage = rewards - predicted_values

    actor_loss = actor.fit([board_obs, scalar_obs, ship_select_obs, advantage, preds],
                            [actions],
                            batch_size=1,
                            shuffle=True,
                            epochs=3,
                            verbose=False)

    critic_loss = critic.fit([board_obs, scalar_obs],
                             [rewards],
                             batch_size=16,
                             shuffle=True,
                             epochs=3,
                             verbose=False)

    actor_loss_collect.extend(actor_loss.history['loss'][-5:])
    critic_loss_collect.extend(critic_loss.history['loss'][-5:])

    if ep_counter % 30 == 0:
        with open('output/episode_scores.txt', 'a+') as f:
            for e in ep_reward_collect:
                f.write(str(e) + '\n')

        with open('output/actor_loss.txt', 'a+') as f:
            for e in actor_loss_collect:
                f.write(str(e) + '\n')

        with open('output/critic_loss.txt', 'a+') as f:
            for e in critic_loss_collect:
                f.write(str(e) + '\n')

        ep_reward_collect = []
        actor_loss_collect = []
        critic_loss_collect = []
