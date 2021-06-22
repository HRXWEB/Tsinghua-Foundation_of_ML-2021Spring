import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="ViZDoom DDDQN parameters")

    parser.add_argument("--model_path", type=str, default="./saves/model", help="Path to save models")
    parser.add_argument("--plot_path", type=str, default="./saves/plot", help="Path to save plot")
    parser.add_argument("--gif_path", type=str, default="./saves/player_gifs", help="Path to save playing agent gifs")
    parser.add_argument("--load_model", action="store_true", help="Either to load model or not")
    parser.add_argument("--max_episodes", type=int, default=1600, help="Maximum episodes per worker")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--play", action="store_true", help="Launch agent to play")
    parser.add_argument("--play_episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--freq_plot", type=int, default=200, help="Frequence of episodes to save plot")
    parser.add_argument("--custom_reward", action="store_true", help="Add penalty if facing backward")

    game_args, _ = parser.parse_known_args()

    return game_args
