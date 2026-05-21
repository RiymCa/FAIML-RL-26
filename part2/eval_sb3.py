import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
import panda_gym  # noqa: F401 - required so Panda envs are registered

def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str, algo_class, base_seed: int = 42,) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    model = algo_class.load(model_path)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        current_seed = base_seed + episode
        obs, info = env.reset(seed=current_seed)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action,_ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO/SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO/SAC model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50,
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        choices=[True, False],
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
    )
