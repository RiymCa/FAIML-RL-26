import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import time
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        choices=[10000, 100000, 500000, 1000000],
        # 1.000 -> 40 s
        # 1.000.000 ->
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    time_start = time.time()
    args = parse_args()
    render = False

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type=args.env_type,
        reward_type="dense",
    )
    env = RandomizationWrapper(env, mode=args.sampling_strategy)

    model = PPO("MultiInputPolicy", env, verbose=0)
    model.learn(total_timesteps=args.timesteps)

    time_end = time.time()
    print(f"time passed -> {int((time_end - time_start)/3600)}:{int((time_end - time_start)/60)}:{(time_end - time_start) % 60:.2f}")

    save_name = f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: model.save(save_name)

    env.close()


if __name__ == "__main__":
    main()
