import argparse
import gymnasium as gym
import time
import os
import panda_gym

from stable_baselines3 import PPO
# Wrapper for base env, done so that it works with both 'none' 'udr'/'adr'
from rand_wrapper import RandomizationWrapper

# Monitor is a wrapper for our environment on top on randomWrapper, its role it's to keep track of the true score of
# the models, kinda acts as a safeguard preventing any errors in the update of his parameters.
# Like a black box on a plane.
from stable_baselines3.common.monitor import Monitor

# Function to evaluate a model, refer to the file to understand its mechanism
from eval_sb3 import evaluate

# - SubprocVecEnv is an environment manager, its role is to take a single env of a model and multiply it,
# parallelizing the model training improving efficiency.
# - DummyVecEnv is an environment manager, since the model, after training, is tested on a single env, it makes that
# env a vectorized one, such as SubProcEnv but with only 1 sub env. Necessary for testing for how our model is built.
# NOTE: we don't use SubprocVecEnv with 1 sub env because it would be incredibly slow since how it's built,
# dummyVecEnv does cover this exact role.
# - VecNormalize is a wrapper for the previous env managers, it keeps track of every observation from our model,
# scaling them to a normal distribution. Gets data from training and applies it in testing.
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# SyncEvalCallback is a function that, refer to the file to understand better
from custom_callback import SyncEvalCallback


"""
    Function for the parsing of the arguments, you can choose any argument you want prior to running with
    default=value.
    The only difference between PPO and SAC is that PPO can use better the parallelization since it is an in place
    algorithm, instead SAC having a buffer to look into it would cause some RAM error if they are too many.
    Also SAC requires many timesteps less than PPO.
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multiple PPO Models on PandaPush-v3")
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
        "--render",
        default=False,
        choices=[True, False],
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        choices=[1000, 500_000, 2_500_000, 10_000_000],
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="Number of parallel environments (CPUs) to use for training",
    )
    return parser.parse_args()


"""
    Function for creating the different environments doing env -> randomWrapper -> Monitor needed for the parallel
    processing for optimizing the training of the PPO models.
"""
def make_env(env_type: str, sampling_strategy: str, rank: int, seed: int = 42):
    def _init() -> gym.Env:
        env = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            type=env_type,
            reward_type="dense",
        )
        env.reset(seed=seed + rank)
        sim = env.unwrapped.task.sim
        object_body_id = sim._bodies_idx["object"]
        mass = sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]
        wrapped_env = RandomizationWrapper(env, mass, mode=sampling_strategy)
        return Monitor(wrapped_env)
    return _init

"""
    The main works like this:
    - Getting the time, the arguments, the frequencies fir the evalCallBack and setting the destination for the models.
    - Creation of a model (for A, B and C):
        - Creation of a folder for the single model.
        - Creates multiple environments for training efficiency, SubprocVecEnv, and wraps then in VecNormalize that
        manages them, you can hover over its hyperparameters to see what they are about.
        - Creates single testing environment with dummyVecEnv to be managed by VecNormalize and set its mode
        'training' to false. 
        - Creates its callBack function, again refer to custom_callback to understand it better.
        - Creates the model, setting device auto -> GPU if available otherwise CPU, and starts the training. Finally
        closes both the training env and the eval env.
    - After all the training of the 3 models we have their best version according to the different hyperparameters,
    we can now print some time stats and start to evaluate them to see which is best.
"""
def main() -> None:
    start_time = time.time()
    args = parse_args()
    eval_freq = max(20_000 // args.num_cpus, 1)
    models_ppo_dir = "./best_models_PPO"
    os.makedirs(models_ppo_dir, exist_ok=True)

    # -------
    # MODEL A
    # -------
    dir_model_a = os.path.join(models_ppo_dir, f"modelA_{args.sampling_strategy}_{args.env_type}")
    print(f"Starting PPO Model A training for {args.timesteps} steps on {args.num_cpus} CPUs.")

    env_a = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, i) for i in range(args.num_cpus)])
    env_a = VecNormalize(env_a, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_a = DummyVecEnv([make_env(args.env_type, args.sampling_strategy, 0)])
    eval_env_a = VecNormalize(eval_env_a, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_a.training = False

    callback_a = SyncEvalCallback(
        eval_env=eval_env_a,
        vec_env=env_a,
        best_model_save_path=dir_model_a,
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
    )

    model_a = PPO("MultiInputPolicy", env_a, device="auto", seed=42, verbose=0)
    print(f"Device in uso: {model_a.device}.")
    model_a.learn(total_timesteps=args.timesteps, callback=callback_a)

    env_a.close()
    eval_env_a.close()

    # -------
    # MODEL B
    # -------
    dir_model_b = os.path.join(models_ppo_dir, f"modelB_{args.sampling_strategy}_{args.env_type}")
    print(f"\nStarting PPO Model B training for {args.timesteps} steps on {args.num_cpus} CPUs.")
    env_b = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, i) for i in range(args.num_cpus)])
    env_b = VecNormalize(env_b, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_b = DummyVecEnv([make_env(args.env_type, "none", 0)])
    eval_env_b = VecNormalize(eval_env_b, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_b.training = False

    callback_b = SyncEvalCallback(
        eval_env=eval_env_b,
        vec_env=env_b,
        best_model_save_path=dir_model_b,
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
    )

    model_b = PPO(
        "MultiInputPolicy",
        env_b,
        device="auto",
        seed=42,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.01,
        verbose=0
    )
    print(f"Device in use: {model_b.device}.")
    model_b.learn(total_timesteps=args.timesteps, callback=callback_b)

    env_b.close()
    eval_env_b.close()

    # -------
    # MODEL C
    # -------
    dir_model_c = os.path.join(models_ppo_dir, f"modelC_{args.sampling_strategy}_{args.env_type}")
    print(f"\nStarting PPO Model C training for {args.timesteps} steps on {args.num_cpus} CPUs.")
    env_c = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, i) for i in range(args.num_cpus)])
    env_c = VecNormalize(env_c, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_c = DummyVecEnv([make_env(args.env_type, "none", 0)])
    eval_env_c = VecNormalize(eval_env_c, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_c.training = False

    callback_c = SyncEvalCallback(
        eval_env=eval_env_c,
        vec_env=env_c,
        best_model_save_path=dir_model_c,
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
    )

    model_c = PPO(
        "MultiInputPolicy",
        env_c,
        device="auto",
        seed=42,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=20,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0
    )
    print(f"Device in use: {model_c.device}.")
    model_c.learn(total_timesteps=args.timesteps, callback=callback_c)

    env_c.close()
    eval_env_c.close()

    # -------
    # TIME AND EVALUATION
    # -------
    end_time = time.time()
    tot_sec = end_time - start_time
    hours = int(tot_sec // 3600)
    minutes = int((tot_sec % 3600) // 60)
    seconds = tot_sec % 60
    print(f"\nTotal time spent: {hours:02d}:{minutes:02d}:{seconds:05.2f}.")

    path_best_a = os.path.join(dir_model_a, "best_model.zip")
    path_best_b = os.path.join(dir_model_b, "best_model.zip")
    path_best_c = os.path.join(dir_model_c, "best_model.zip")

    print("\nEvaluating PPO model A.")
    evaluate(model_path=path_best_a, stats_path=os.path.join(dir_model_a, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)
    print("\nEvaluating PPO model B.")
    evaluate(model_path=path_best_b, stats_path=os.path.join(dir_model_b, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)
    print("\nEvaluating PPO model C.")
    evaluate(model_path=path_best_c, stats_path=os.path.join(dir_model_c, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)

if __name__ == "__main__":
    main()