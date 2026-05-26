import argparse
import gymnasium as gym
import time
import os
import panda_gym

from stable_baselines3 import PPO, SAC
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

# 1 = PPO e SAC both on Source -> Test both on Source e Target
# 2 = 2 SAC models respectively on Source e Target -> Test both on Source and Target
# 3 = 2 SAC models on Source, one with UDR and the other one with ADR -> Test both on Source and Target
CONFIG = 1

"""
    Function for the parsing of the arguments, you can choose any argument you want prior to running with
    default=value.
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=250_000,
        help="Timestep per ciascun modello"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=8,
        help="Numero di CPU per il parallelismo"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Numero di episodi fissi di valutazione"
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


"""
    Function to train either a PPO or SAC model with best hyperparameter tuning found earlier, also frees RAM after end.
"""
def train_agent(model_name: str, algo_class, env_type: str, sampling_strategy: str, timesteps: int, num_cpus: int):
    print(f"Starting training: {model_name}")
    print(f"Domain: {env_type.upper()} | Randomization: {sampling_strategy.upper()}")

    if CONFIG == 1:
        models_dir = "./final_models_task4"
    elif CONFIG == 2:
        models_dir = "./final_models_task5"
    else:
        models_dir = "./final_models_task6"

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

    if algo_class == PPO:
        model = PPO(
            "MultiInputPolicy",
            env,
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
    elif algo_class == SAC:
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
    else:
        raise ValueError("Algoritmo non supportato. Scegliere tra PPO e SAC.")

    model.learn(total_timesteps=timesteps, callback=callback)

    env.close()
    eval_env.close()

    end_time = time.time()
    diff = end_time - start_time
    ore = int(diff // 3600)
    minuti = int((diff % 3600) // 60)
    print(f"{model_name} training completato in {ore}h {minuti}m.")

    model_path = os.path.join(dir_model, "best_model.zip")
    stats_path = os.path.join(dir_model, "vec_normalize.pkl")

    del model, env, eval_env, callback
    return model_path, stats_path


"""
    Given a global variable 'CONFIG' the main executes different path according to which task we want to complete.
"""
def main() -> None:
    args = parse_args()

    if CONFIG == 1:
        # -----------------------------
        # PPO vs SAC trained on Source
        # -----------------------------
        print("Task 4: PPO vs SAC trained on Source and tested on Source and Target")

        path_ppo, stats_ppo = train_agent("PPO_Source", PPO, "source", "none", args.timesteps*50, args.num_cpus*2)
        path_sac, stats_sac = train_agent("SAC_Source", SAC, "source", "none", args.timesteps, args.num_cpus)

        print(f"\nEvaluation results with {args.eval_episodes} episodes")

        print("PPO")
        print("Test on Source")
        evaluate(model_path=path_ppo, stats_path=stats_ppo, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=PPO)
        print("Test on Target")
        evaluate(model_path=path_ppo, stats_path=stats_ppo, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=PPO)

        print("\nSAC")
        print("Test on Source")
        evaluate(model_path=path_sac, stats_path=stats_sac, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("Test on Target")
        evaluate(model_path=path_sac, stats_path=stats_sac, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

    elif CONFIG == 2:
        # -----------------------------------------------------------
        # SAC on Source vs SAC on Target tested on Source and Target
        # -----------------------------------------------------------
        print("Task 5: SAC on Source vs SAC on Target")

        path_sac_source, stats_sac_source = train_agent("SAC_Source", SAC, "source", "none", args.timesteps, args.num_cpus)
        path_sac_target, stats_sac_target = train_agent("SAC_Target", SAC, "target", "none", args.timesteps, args.num_cpus)

        print(f"\nEvaluation results with {args.eval_episodes} episodes")

        print("SAC on Source")
        print("Test on Source")
        evaluate(model_path=path_sac_source, stats_path=stats_sac_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("Test on Target")
        evaluate(model_path=path_sac_source, stats_path=stats_sac_source, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

        print("SAC on Target")
        print("Test on Target")
        evaluate(model_path=path_sac_target, stats_path=stats_sac_target, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

    elif CONFIG == 3:
        # -------------------------------------------------------
        # SAC with UDR vs SAC with ADR on both Source and Target
        # -------------------------------------------------------
        print("SAC with UDR vs SAC with ADR on both Source and Target")

        path_udr, stats_udr = train_agent("SAC_UDR_Source", SAC, "source", "udr", args.timesteps, args.num_cpus)
        path_adr, stats_adr = train_agent("SAC_ADR_Source", SAC, "source", "adr", args.timesteps, args.num_cpus)

        print(f"\nEvaluation results with {args.eval_episodes} episodes")

        print("SAC UDR on Source")
        print("Test on source")
        evaluate(model_path=path_udr, stats_path=stats_udr, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("Test on Target")
        evaluate(model_path=path_udr, stats_path=stats_udr, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)

        print("SAC ADR on Source")
        print("Test on Source")
        evaluate(model_path=path_adr, stats_path=stats_adr, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="source", algo_class=SAC)
        print("Test on Target")
        evaluate(model_path=path_adr, stats_path=stats_adr, n_episodes=args.eval_episodes,
                 deterministic=True, render=False, env_type="target", algo_class=SAC)


if __name__ == "__main__":
    main()
