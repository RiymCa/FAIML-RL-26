import argparse
import gymnasium as gym
import time
import os
import panda_gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from rand_wrapper import RandomizationWrapper
from eval_sb3 import evaluate
from custom_callback import SyncEvalCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multiple SAC Models on PandaPush-v3")
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
        default=1_000_000,
        choices=[1000, 100_000, 500_000, 1_000_000, 2_500_000],
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
        default=8,
        help="Number of parallel environments (CPUs) to use for training",
    )
    return parser.parse_args()


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


def main() -> None:
    start_time = time.time()
    args = parse_args()

    eval_freq = max(20_000 // args.num_cpus, 1)
    models_sac_dir = "./best_models_SAC"
    os.makedirs(models_sac_dir, exist_ok=True)

    # -------
    # MODEL A
    # -------
    dir_model_a = os.path.join(models_sac_dir, f"sac_modelA_{args.sampling_strategy}_{args.env_type}")
    print(f"Starting SAC Model A training for {args.timesteps} steps on {args.num_cpus} CPUs.")

    env_a = SubprocVecEnv([make_env(args.env_type, args.sampling_strategy, i) for i in range(args.num_cpus)])
    env_a = VecNormalize(env_a, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_a = DummyVecEnv([make_env(args.env_type, "none", 0)])
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

    model_a = SAC("MultiInputPolicy", env_a, device="auto", seed=42, verbose=0)
    print(f"Device in use: {model_a.device}.")
    model_a.learn(total_timesteps=args.timesteps, callback=callback_a)

    env_a.close()
    eval_env_a.close()

    end_time_1 = time.time()
    diff1 = end_time_1 - start_time
    print(f"Time spent model A: {int(diff1 // 3600)}:{int((diff1 % 3600) // 60)}:{(diff1 % 60):.2f}")

    # Freeing ram usage
    del model_a, env_a, eval_env_a, callback_a

    # -------
    # MODEL B
    # -------
    dir_model_b = os.path.join(models_sac_dir, f"sac_modelB_{args.sampling_strategy}_{args.env_type}")
    print(f"\nStarting SAC Model B training for {args.timesteps} steps on {args.num_cpus} CPUs.")

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

    model_b = SAC(
        "MultiInputPolicy",
        env_b,
        device="auto",
        seed=42,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=0
    )
    print(f"Device in use: {model_b.device}.")
    model_b.learn(total_timesteps=args.timesteps, callback=callback_b)
    env_b.close()
    eval_env_b.close()

    end_time_2 = time.time()
    diff2 = end_time_2 - end_time_1
    print(f"Time spent model B: {int(diff2 // 3600)}:{int((diff2 % 3600) // 60)}:{(diff2 % 60):.2f}")

    # Freeing ram usage
    del model_b, env_b, eval_env_b, callback_b

    # -------
    # MODEL C
    # -------
    dir_model_c = os.path.join(models_sac_dir, f"sac_modelC_{args.sampling_strategy}_{args.env_type}")
    print(f"\nStarting SAC Model C training for {args.timesteps} steps on {args.num_cpus} CPUs.")

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

    model_c = SAC(
        "MultiInputPolicy",
        env_c,
        device="auto",
        seed=42,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=512,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=args.num_cpus,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=0
    )
    print(f"Device in use: {model_c.device}.")
    model_c.learn(total_timesteps=args.timesteps, callback=callback_c)
    env_c.close()
    eval_env_c.close()

    end_time_3 = time.time()
    diff3 = end_time_3 - end_time_2
    print(f"Time spent model C: {int(diff3 // 3600)}:{int((diff3 % 3600) // 60)}:{(diff3 % 60):.2f}")

    # Freeing ram usage
    del model_c, env_c, eval_env_c, callback_c

    # -------
    # TIME AND EVALUATION
    # -------
    diff_tot = end_time_3 - start_time
    print(f"Total time spent: {int(diff_tot // 3600)}:{int((diff_tot % 3600) // 60)}:{(diff_tot % 60):.2f}")

    path_best_a = os.path.join(dir_model_a, "best_model.zip")
    path_best_b = os.path.join(dir_model_b, "best_model.zip")
    path_best_c = os.path.join(dir_model_c, "best_model.zip")

    print("\nEvaluating SAC model A.")
    evaluate(model_path=path_best_a, stats_path=os.path.join(dir_model_a, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=SAC)
    print("\nEvaluating SAC model B.")
    evaluate(model_path=path_best_b, stats_path=os.path.join(dir_model_b, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=SAC)
    print("\nEvaluating SAC model C.")
    evaluate(model_path=path_best_c, stats_path=os.path.join(dir_model_c, "vec_normalize.pkl"), n_episodes=100,
             deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=SAC)

if __name__ == "__main__":
    main()