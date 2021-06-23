from train import train
from eval import evaluate


if __name__ == "__main__":

    skip_training = True
    load_model = False
    play_episodes = 5

    if skip_training:
        evaluate(play_episodes)
    else:
        train(load_model)
