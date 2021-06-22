import numpy as np
import os, sys
from torchvision.io import write_video

from dddqn_agent import DDDQNAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import  make_envEVAL


def evaluate(num_of_games):
    env = make_envEVAL("VizdoomDefendLine-v0")

    agent = DDDQNAgent(
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
        chkpt_dir="./content/",
        algo="DDDQNAgent",
        env_name="vizdoomgym",
    )

    img_array = []

    agent.load_models()

    for i in range(num_of_games):
        done = False
        observation, obs = env.reset()
        img_array.append(obs)

        score = 0
        while not done:

            action = agent.choose_action(observation)
            (observation, obs2), reward, done, _ = env.step(action)
            score += reward
            for _ in range(2):
                img_array.append(obs2)

        for _ in range(12):
            img_array.append(np.empty([240, 320, 3], dtype=np.uint8))

        print("Episode #", i, " Rewards: ", score)

    # Writes the the output image sequences in a video file
    write_video("./content/doom.mp4", img_array, 25, video_codec="libx264", options=None)
