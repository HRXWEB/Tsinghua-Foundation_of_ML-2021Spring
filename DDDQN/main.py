from train import train
from eval import evaluate
from utils.args import parse_arguments


if __name__ == "__main__":

    params = parse_arguments()

    if params.play:
        evaluate(params.play_episodes)
    else:
        train(params.load_model)
