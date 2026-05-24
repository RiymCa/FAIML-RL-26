import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import time
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC
from rand_wrapper import RandomizationWrapper
from eval_sb3 import evaluate

np.random.seed(42)

def parse_args(var: bool) -> argparse.Namespace:
    if var:
        st1 = st2 = "none"
        et2 = "target"
    else:
        st1 = "udr"
        st2 = "adr"
        et2 = "source"

    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy_1",
        type=str,
        default=st1,
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--sampling-strategy_2",
        type=str,
        default=st2,
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type_1",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--env-type_2",
        type=str,
        default=et2,
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        choices=[1000, 100_000, 500_000, 1_000_000, 2_500_000],
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        default=False,
        choices=[True, False],
        help="Render with a window (render_mode='human')",
    )

    return parser.parse_args()


def make_train_env(env_type: str, sampling_strategy: str, render: bool) -> gym.Env:
    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type=env_type,
        reward_type="dense",
    )
    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]
    mass = sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]

    return RandomizationWrapper(env, mass, mode=sampling_strategy, env_type=env_type, verbose=True)


def main() -> None:
    # Variable to decide whether we are doing part 3 or part 4.
    # True if we are doing part 3.
    # False if we are doing part 4.
    var = True
    args = parse_args(var)
    print("Creating model 1.")
    time_start = time.time()

    env_1 = make_train_env(args.env_type_1, args.sampling_strategy_1, args.render)
    model_1 = SAC("MultiInputPolicy", env_1, device="cuda", seed=42, verbose=0)
    print(f"Device in use: {model_1.device}")
    model_1.learn(total_timesteps=args.timesteps)

    time_end_1 = time.time()
    diff_1 = time_end_1 - time_start
    print(f"Finished model 1 training.\n"
          f"Time passed -> {int(diff_1 // 3600):02d}:{int((diff_1 % 3600) // 60):02d}:{(diff_1 % 60):.2f}\n")

    env_1.close()

    print("Creating model 2.")
    env_2 = make_train_env(args.env_type_2, args.sampling_strategy_2, args.render)
    model_2 = SAC("MultiInputPolicy", env_2, device="cuda", seed=42, verbose=0)
    print(f"Device in use: {model_2.device}")
    model_2.learn(total_timesteps=args.timesteps)

    time_end_2 = time.time()
    diff_2 = time_end_2 - time_end_1
    print(f"Finished model 2 training.\n"
          f"Time passed -> {int(diff_2 // 3600):02d}:{int((diff_2 % 3600) // 60):02d}:{(diff_2 % 60):.2f}\n")

    env_2.close()

    diff_tot = time_end_2 - time_start
    print(f"Total time passed -> {int(diff_tot // 3600):02d}:{int((diff_tot % 3600) // 60):02d}:{(diff_tot % 60):.2f}")

    save_name_1 = f"sac_push_model1_{args.sampling_strategy_1}_{args.env_type_1}_{args.timesteps // 1000}k"
    save_name_2 = f"sac_push_model2_{args.sampling_strategy_2}_{args.env_type_2}_{args.timesteps // 1000}k"
    model_1.save(save_name_1)
    model_2.save(save_name_2)

    print("\nEvaluating model 1.")
    evaluate(model_path=f"{save_name_1}.zip",
             n_episodes=50,
             deterministic=not args.stochastic,
             render=args.render,
             env_type=args.env_type_1,
             algo_class=SAC)

    print("\nEvaluating model 2.")
    evaluate(model_path=f"{save_name_2}.zip",
             n_episodes=50,
             deterministic=not args.stochastic,
             render=args.render,
             env_type=args.env_type_2,
             algo_class=SAC)



if __name__ == "__main__":
    main()