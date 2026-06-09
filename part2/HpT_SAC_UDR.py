import argparse
import time
import gymnasium as gym
import argparse
import gymnasium as gym
import time
import os
import panda_gym

from stable_baselines3 import SAC
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
# NOTE: we don't use SubprocVecEnv with 1 sub env because it would be incredibly slow since how
# it's built, dummyVecEnv does cover this exact role.
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
    parser = argparse.ArgumentParser(description="Tune UDR Mass Ranges for SAC Models on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="udr",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type to train on",
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
        choices=[1000, 250_000, 500_000, 1_000_000],
        help="Number of training timesteps per model",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=8,
        help="Number of parallel environments (CPUs) to use for training",
    )
    return parser.parse_args()


"""
    Function for creating the different environments doing env -> randomWrapper -> Monitor, needed for the parallel
    processing for optimizing the training of the SAC models.
"""
def make_env(env_type: str, sampling_strategy: str, mass_range: tuple, rank: int, seed: int = 42):
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

        is_verbose = (rank == 0)

        wrapped_env = RandomizationWrapper(env, mass, mass_range=mass_range, mode=sampling_strategy, verbose=is_verbose)
        return Monitor(wrapped_env)

    return _init


"""
    The main works like this:
    - Getting the time, the arguments, the frequencies fir the evalCallBack and setting the destination for the models.
    - Creation of the same three models (for A, B and C) it only changes the range they are trained in with udr:
        - Creation of a folder for the single model.
        - Creates multiple environments for training efficiency, SubprocVecEnv, and wraps then in VecNormalize that
        manages them, you can hover over its hyperparameters to see what they are about.
        - Creates single testing environment with dummyVecEnv to be managed by VecNormalize and set its mode
        'training' to false. 
        - Creates its callBack function, again refer to custom_callback to understand it better.
        - Creates the model, setting device auto -> GPU if available otherwise CPU, and starts the training. Finally
        closes both the training env and the eval env.
    - After all the training of the 3 models we have their best version according to the different mass ranges tested,
    we can now print some time stats and start to evaluate them to see which is best.
    
    Effectively this file trains the same models with three different versions of mass ranges of udr, to see which
    training allows the model to perform better both in source and in training.
"""
def main() -> None:
    start_time = time.time()
    args = parse_args()

    eval_freq = max(20_000 // args.num_cpus, 1)
    models_udr_dir = "./best_models_UDR"
    os.makedirs(models_udr_dir, exist_ok=True)
    test_env_type = "target"

    # -------
    # MODEL A : [0.5, 2.0]
    # -------
    range_a = (0.5, 2.0)
    dir_model_a = os.path.join(models_udr_dir, f"sac_UDR_A_{range_a[0]}-{range_a[1]}")

    print(f"Starting SAC UDR Model A training, Range: {range_a} for {args.timesteps} steps on {args.num_cpus} CPUs.")

    env_a = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, range_a, i) for i in range(args.num_cpus)])
    env_a = VecNormalize(env_a, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_a = DummyVecEnv([make_env(args.env_type, args.sampling_strategy, range_a, 0)])
    eval_env_a = VecNormalize(eval_env_a, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_a.training = False

    callback_a = SyncEvalCallback(
        eval_env=eval_env_a,
        vec_env=env_a,
        best_model_save_path=dir_model_a,
        n_eval_episodes=20,
        eval_freq=eval_freq,
        deterministic=True,
    )
    model_a = SAC(
        "MultiInputPolicy",
        env_a,
        device="auto",
        seed=42,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gradient_steps=args.num_cpus,
        verbose=0,
    )
    model_a.learn(total_timesteps=args.timesteps, callback=callback_a)

    env_a.close()
    eval_env_a.close()

    end_time_1 = time.time()
    print(f"Time spent model A: {int((end_time_1 - start_time) // 3600)}:{int(((end_time_1 - start_time) % 3600) // 60)}:{((end_time_1 - start_time) % 60):.2f}")

    del model_a, env_a, eval_env_a, callback_a

    # -------
    # MODEL B : [0.5, 6.0]
    # -------
    range_b = (0.5, 6.0)

    dir_model_b = os.path.join(models_udr_dir, f"sac_UDR_B_{range_b[0]}-{range_b[1]}")
    print(f"\nStarting SAC UDR Model B training, Range: {range_b} for {args.timesteps} steps on {args.num_cpus} CPUs.")

    env_b = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, range_b, i) for i in range(args.num_cpus)])
    env_b = VecNormalize(env_b, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_b = DummyVecEnv([make_env(args.env_type, args.sampling_strategy, range_b, 0)])
    eval_env_b = VecNormalize(eval_env_b, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_b.training = False

    callback_b = SyncEvalCallback(
        eval_env=eval_env_b,
        vec_env=env_b,
        best_model_save_path=dir_model_b,
        n_eval_episodes=20,
        eval_freq=eval_freq,
        deterministic=True,
    )
    model_b = SAC(
        "MultiInputPolicy",
        env_b,
        device="auto",
        seed=123,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gradient_steps=args.num_cpus,
        verbose=0
    )
    model_b.learn(total_timesteps=args.timesteps, callback=callback_b)

    env_b.close()
    eval_env_b.close()

    end_time_2 = time.time()
    print(f"Time spent model B: {int((end_time_2 - end_time_1) // 3600)}:{int(((end_time_2 - end_time_1) % 3600) // 60)}:{((end_time_2 - end_time_1) % 60):.2f}")

    del model_b, env_b, eval_env_b, callback_b

    # -------
    # MODEL C : [0.1, 10.0]
    # -------
    range_c = (0.1, 10.0)

    dir_model_c = os.path.join(models_udr_dir, f"sac_UDR_C_{range_c[0]}-{range_c[1]}")
    print(f"\nStarting SAC UDR Model C training, Range: {range_c} for {args.timesteps} steps on {args.num_cpus} CPUs.")

    env_c = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, range_c, i) for i in range(args.num_cpus)])
    env_c = VecNormalize(env_c, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_c = DummyVecEnv([make_env(args.env_type, args.sampling_strategy, range_c, 0)])
    eval_env_c = VecNormalize(eval_env_c, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env_c.training = False

    callback_c = SyncEvalCallback(
        eval_env=eval_env_c,
        vec_env=env_c,
        best_model_save_path=dir_model_c,
        n_eval_episodes=20,
        eval_freq=eval_freq,
        deterministic=True,
    )
    model_c = SAC(
        "MultiInputPolicy",
        env_c,
        device="auto",
        seed=42,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        gradient_steps=args.num_cpus,
        verbose=0,
    )
    model_c.learn(total_timesteps=args.timesteps, callback=callback_c)

    env_c.close()
    eval_env_c.close()

    end_time_3 = time.time()
    print(f"Time spent model C: {int((end_time_3 - end_time_2) // 3600)}:{int(((end_time_3 - end_time_2) % 3600) // 60)}:{((end_time_3 - end_time_2) % 60):.2f}")

    del model_c, env_c, eval_env_c, callback_c

    # -------
    # TIME AND EVALUATION
    # -------
    diff_tot = end_time_3 - start_time
    print(f"\nTotal time spent: {int(diff_tot // 3600)}:{int((diff_tot % 3600) // 60)}:{(diff_tot % 60):.2f}")

    dirs = {"A": dir_model_a, "B": dir_model_b, "C": dir_model_c}
    ranges = {"A": range_a, "B": range_b, "C": range_c}

    for label in ["A", "B", "C"]:
        path = os.path.join(dirs[label], "best_model.zip")
        stats = os.path.join(dirs[label], "vec_normalize.pkl")

        print(f"Evaluating SAC UDR model {label} Range {ranges[label]}")

        print(f"\nTest on SOURCE")
        evaluate(
            model_path=path,
            stats_path=stats,
            n_episodes=100,
            deterministic=not args.stochastic,
            render=args.render,
            env_type=args.env_type,
            algo_class=SAC
        )

        print(f"\nTest on TARGET")
        evaluate(
            model_path=path,
            stats_path=stats,
            n_episodes=100,
            deterministic=not args.stochastic,
            render=args.render,
            env_type=test_env_type,
            algo_class=SAC
        )


if __name__ == "__main__":
    main()