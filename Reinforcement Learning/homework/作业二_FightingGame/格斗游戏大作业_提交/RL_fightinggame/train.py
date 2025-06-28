'''
Created by WenYaozhi GuShaowei PanYuqi
'''

import os
import argparse
from fightingice_env import FightingiceEnv

from methods import SARSA
from methods import DQN as DQN
from methods import QLearning


def parse_args():
    parser = argparse.ArgumentParser(description="FightingICE game")
    parser.add_argument("--method",
                        type=str,
                        default="DQN",
                        choices=["SARSA", "QLearning", "DQN"],
                        help="Choose which method to use.")
    parser.add_argument("--player", type=str,
                        default="ZEN",
                        choices=["ZEN", "LUD", "GARNET"],
                        help="Choose which character to fight.")
    parser.add_argument("--save_path", type=str,
                        default="./checkpoints",
                        help="Choose where to save the model.")
    parser.add_argument("--log_path", type=str,
                        default="./train_log",
                        help="Choose where to save the log file.")
    #Train parameters
    parser.add_argument("--epochs", type=int,
                        default=300,
                        help="Choose how many epochs to train.")
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="Choose the batch size.")


    args = parser.parse_args()
    return args


def main(args):
    """
    Main function.
    """
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    env = FightingiceEnv(character=args.player, port=4242)
    # env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute", "-r", "100"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]

    ckpt_save_root = os.path.join(
        os.path.join(args.save_path, args.method), args.player
    )
    if not os.path.exists(ckpt_save_root):
        os.makedirs(ckpt_save_root)

    # 初始化模型
    if args.method == "SARSA":
        model = SARSA(environment=env, environment_args=env_args)
    elif args.method == "QLearning":
        model = QLearning(args=args, environment=env, environment_args=env_args)
    elif args.method == "DQN":
        model = DQN(args=args, environment=env, environment_args=env_args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    # 训练模型
    model.Train(args=args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
