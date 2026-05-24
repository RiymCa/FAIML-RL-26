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
# NOTE: we don't use SubprocVecEnv with 1 sub env because it would be incredibly slow since how it's built,
# dummyVecEnv does cover this exact role.
# - VecNormalize is a wrapper for the previous env managers, it keeps track of every observation from our model,
# scaling them to a normal distribution. Gets data from training and applies it in testing.
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# SyncEvalCallback is a function that, refer to the file to understand better
from custom_callback import SyncEvalCallback

# False = only does part 3.2 or task 5
# True  = only does part 4 or task 6
MAKE_TASK_6 = False


"""
    Function for the parsing of the arguments, you can choose any argument you want prior to running with
    default=value.
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Timesteps per each model"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=8,
        help="CPUs for SubprocVecEnv"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of test episodes"
    )
    return parser.parse_args()


"""
    Function for creating the different environments doing env -> randomWrapper -> Monitor needed for the parallel
    processing for optimizing the training of the SAC models.
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


def train_agent(model_name: str, env_type: str, sampling_strategy: str, timesteps: int, num_cpus: int):
    print(f"Starting training: {model_name}")
    print(f"Domain: {env_type.upper()} | Randomization: {sampling_strategy.upper()}")

    if MAKE_TASK_6:
        models_dir = "./final_models_task6"
    else:
        models_dir = "./final_models_task5"
    start_time = time.time()
    os.makedirs(models_dir, exist_ok=True)
    dir_model = os.path.join(models_dir, model_name)
    eval_freq = max(20_000 // num_cpus, 1)

    env = SubprocVecEnv([make_env(env_type, sampling_strategy, i) for i in range(num_cpus)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = DummyVecEnv([make_env(env_type, "none", 0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    callback = SyncEvalCallback(
        eval_env=eval_env,
        vec_env=env,
        best_model_save_path=dir_model,
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        device="auto",
        seed=42,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=512,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=num_cpus,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=0
    )

    model.learn(total_timesteps=timesteps, callback=callback)

    env.close()
    eval_env.close()

    end_time = time.time()
    diff = end_time - start_time
    print(f"{model_name} training completed in {int(diff // 3600)}h {int((diff % 3600) // 60)}m.")

    model_path = os.path.join(dir_model, "best_model.zip")
    stats_path = os.path.join(dir_model, "vec_normalize.pkl")

    del model, env, eval_env, callback
    return model_path, stats_path


def main() -> None:
    args = parse_args()

    if not MAKE_TASK_6:
        # ------
        # Task 5
        # ------
        print("Task 5.")

        path_base_source, stats_base_source = train_agent("task_5_source", "source",
                                                          "none", args.timesteps, args.num_cpus)
        path_base_target, stats_base_target = train_agent("task_5_target", "target",
                                                          "none", args.timesteps,                                                          args.num_cpus)

        print("\nEvaluation of the models.")

        print("Model A, based on source, tested on source")
        evaluate(model_path=path_base_source, stats_path=stats_base_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)

        print("Model A, based on source, tested on target")
        evaluate(model_path=path_base_source, stats_path=stats_base_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

        print("Model B, based on target, tested on target")
        evaluate(model_path=path_base_target, stats_path=stats_base_target, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

    else:
        # -------
        # task 6
        # -------
        print("Task 6.")

        path_udr_source, stats_udr_source = train_agent("UDR_Source", "source", "udr",
                                                        args.timesteps, args.num_cpus)
        path_adr_source, stats_adr_source = train_agent("ADR_Source", "source", "adr",
                                                        args.timesteps, args.num_cpus)

        print("\nEvaluation of the models.")

        print("UDR model, test on source")
        evaluate(model_path=path_udr_source, stats_path=stats_udr_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("UDR model, test on target")
        evaluate(model_path=path_udr_source, stats_path=stats_udr_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

        print("ADR model, test on source")
        evaluate(model_path=path_adr_source, stats_path=stats_adr_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("ADR model, test on target")
        evaluate(model_path=path_adr_source, stats_path=stats_adr_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)


if __name__ == "__main__":
    main()