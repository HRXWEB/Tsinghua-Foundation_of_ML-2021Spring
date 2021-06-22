import numpy as np
import os, sys

from dddqn_agent import DDDQNAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import make_env, plot_learning_curve, plot_loss_curve


def train(load_checkpoint):
    env = make_env("VizdoomDefendLine-v0")
    best_score = -np.inf
    n_games = 1500
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

    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_array, loss_array = [], [], [], []
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward

            agent.store_transition(observation, action, reward, observation_, int(done))
            loss = agent.learn()
            if not loss is None:
                loss_array.append(loss)

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score

            print("Checkpoint saved at episode ", i)
            agent.save_models()

        print(
            "Episode: ",
            i,
            "Score: ",
            score,
            " Average score: %.2f" % avg_score,
            "Best average: %.2f" % best_score,
            "Epsilon: %.2f" % agent.epsilon,
            "Steps:",
            n_steps,
        )

        eps_history.append(agent.epsilon)

        if (i + 1) % 100 == 0:
            fname = agent.algo + "_" + agent.env_name + "_lr" + str(agent.lr) + "_" + str(i + 1) + "games_score"
            figure_file = "plots/" + fname + ".png"
            x = [i + 1 for i in range(len(scores))]
            plot_learning_curve(x, scores, eps_history, figure_file)
            fname = agent.algo + "_" + agent.env_name + "_lr" + str(agent.lr) + "_" + str(i + 1) + "games_loss"
            figure_file = "plots/" + fname + ".png"
            x = [i + 1 for i in range(len(loss_array))]
            plot_loss_curve(x, loss_array, figure_file)
