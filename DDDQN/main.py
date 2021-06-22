from train import train
from eval import evaluate


if __name__ == "__main__":
    load_checkpoint = False
    skip_learning = False

    evaluation = False
    num_of_games = 5

    if not skip_learning:
        train(load_checkpoint)

    if evaluation:
        evaluate(num_of_games)
