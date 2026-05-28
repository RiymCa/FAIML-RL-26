import argparse
import os
import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC, PPO
from rand_wrapper import RandomizationWrapper

# - DummyVecEnv is an environment manager, since the model, after training, is tested on a single env, it makes that
# env a vectorized one.
# - VecNormalize is a wrapper for the previous env managers, it keeps track of every observation from our model,
# scaling them to a normal distribution. Gets data from training and applies it in testing.
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


"""
    It works like this:
    - Checks if the path and models really exists.
    - Creates the dummyVecEnv and puts it inside the VecNormalize wrapper, loading the stats of the best model we had.
    - We load the model.
    - Starts the evaluation phase, it is based on iterative seeds to provide an equal evaluation env to all the models.
    - Gets the results of the simulations on num_episodes and the n prints them.
"""
def evaluate(model_path: str, stats_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str,
             algo_class, base_seed: int = 42,) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {stats_path}. Ensure you trained with VecNormalize.")
    render_mode = "human" if render else "rgb_array"

    def make_test_env():
        env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
        sim = env.unwrapped.task.sim
        object_body_id = sim._bodies_idx["object"]
        mass = sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]

        return RandomizationWrapper(env, mass, mode="none", env_type=env_type)

    env = DummyVecEnv([make_test_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False # So the stats don't change anymore.
    env.norm_reward = False # So that the outputs gives us unnormalized parameters, but still working with normalized ones.

    model = algo_class.load(model_path)
    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        current_seed = base_seed + episode
        env.seed(current_seed)
        obs = env.reset()

        done = False
        episode_return = 0.0

        while not done:
            action,_ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, info = env.step(action)

            episode_return += float(reward[0])
            done = dones[0]
            if done:
                if "is_success" in info[0]:
                    successes.append(float(info[0]["is_success"]))
                break

        episode_returns.append(episode_return)
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


"""
    Useless since we import the evaluate function into evry file to evaluate in place. 
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO/SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO/SAC model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to vec_normalize.pkl"
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
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"]
    )
    return parser.parse_args()

"""
    Useless since we import the evaluate function into evry file to evaluate in place. 
"""
if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model_path,
        stats_path=args.stats_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        algo_class=args.algo
    )
