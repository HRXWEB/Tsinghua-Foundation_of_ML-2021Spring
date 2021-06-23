import numpy as np
import os

from dqn_agent import DQNAgent
from utils.util import make_envEVAL, make_gif


def evaluate(num_of_games):
    model_path = "./content"
    gif_path = "./gif"
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    env = make_envEVAL("VizdoomDefendLine-v0")

    agent = DQNAgent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.0001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=5000,
        eps_min=0.1,
        batch_size=32,
        replace=1000,
        eps_dec=1e-5,
        chkpt_dir=model_path,
        algo="DQNAgent",
        env_name="vizdoomgym",
    )

    agent.load_models()

    for i in range(num_of_games):
        img_array = []
        done = False
        observation, obs = env.reset()
        img_array.append(obs)
        step = 1
        score = 0
        while not done:
            step += 1
            action = agent.choose_best_action(observation)
            (observation, obs2), reward, done, _ = env.step(action)
            score += reward
            img_array.append(obs2)

        print("Episode #", i, " Rewards: ", score, "Steps: ", step)

        images = np.array(img_array)
        gif_file = os.path.join(gif_path, agent.env_name + "_game_" + str(i + 1) + ".gif")
        make_gif(images, gif_file, fps=100)
