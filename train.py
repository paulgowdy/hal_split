import config as cfg
from halite_environment import HaliteEnvironment
from utils import process_obs

# Create environment
env = HaliteEnvironment(cfg.BOARD_SIZE)
# Reset env
raw_obs = env.reset()
#Process obs




current_board, current_scalars, current_ship_select = process_obs(raw_obs)

# Create Actor
actor_model = 1
# Create Critic
critic_model = 1

for ep in range(cfg.PPO_BATCHES):
    print("Episode", ep)

    # Blank-out PPO memory buffer


    for step in range(cfg.PPO_STEPS):
        # In general, the ppo steps won't line up evenly with the end of an Episode
        # So just collect, train when ready, and reset when done



        raw_obs, reward, done, info = env.step(action)
        #Process obs

        # Build up memory buffer

        if done:
            env.reset()
            # Log episode scores

    # Get last q_value, append it to value memory buffer
    # Why getting this again?

    # Get returns and advantages

    # Actor.model.fit
    # Critic.model.fit

    # Log losses here


    # Every so often, run eval eps
    # Get the current eval score
    # If score is best, save model
