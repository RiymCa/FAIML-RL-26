import argparse
import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rand_wrapper import RandomizationWrapper
from eval_sb3 import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multiple SAC Models on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="adr",
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
        default=500000,
        choices=[10000, 50000, 100000, 200000, 500000, 1000000],
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    return parser.parse_args()


def make_train_env(env_type: str, sampling_strategy: str) -> gym.Env:
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type="dense",
    )
    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]
    mass = sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]

    return RandomizationWrapper(env, mass, mode=sampling_strategy)



def main() -> None:
    start_time = time.time()
    args = parse_args()

    print(f"Starting SAC ADR Model A training for {args.timesteps} steps.")
    env_a = make_train_env(args.env_type, args.sampling_strategy)
    model_a = SAC("MultiInputPolicy", env_a, device="cuda", seed=42, verbose=0)
    print(f"Device in use: {model_a.device}.")
    model_a.learn(total_timesteps=args.timesteps)
    env_a.close()

    end_time_1 = time.time()
    diff1 = end_time_1 - start_time
    print(f"Time spent model A: {int(diff1 // 3600)}:{int((diff1 % 3600) // 60)}:{(diff1 % 60):.2f}")


    print(f"\nStarting SAC ADR Model B training for {args.timesteps} steps.")
    env_b = make_train_env(args.env_type, args.sampling_strategy)
    model_b = SAC(
        "MultiInputPolicy",
        env_b,
        device="cuda",
        seed=42,
        buffer_size=2000000,
        batch_size=512,
        learning_rate=3e-4,
        ent_coef="auto",
        verbose=0
    )
    print(f"Device in use: {model_b.device}.")
    model_b.learn(total_timesteps=args.timesteps)
    env_b.close()

    end_time_2 = time.time()
    diff2 = end_time_2 - end_time_1
    print(f"Time spent model B: {int(diff2 // 3600)}:{int((diff2 % 3600) // 60)}:{(diff2 % 60):.2f}")

    print(f"\nStarting SAC ADR Model C training for {args.timesteps} steps.")
    env_c = make_train_env(args.env_type, args.sampling_strategy)
    model_c = SAC(
        "MultiInputPolicy",
        env_c,
        device="cuda",
        seed=42,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        train_freq=1,
        gradient_steps=2,
        gamma=0.98,
        tau=0.02,
        learning_rate=3e-4,
        ent_coef="auto",
        verbose=0
    )
    print(f"Device in use: {model_c.device}.")
    model_c.learn(total_timesteps=args.timesteps)
    env_c.close()

    end_time_3 = time.time()
    diff3 = end_time_3 - end_time_2
    print(f"Time spent model C: {int(diff3 // 3600)}:{int((diff3 % 3600) // 60)}:{(diff3 % 60):.2f}")

    diff_tot = end_time_3 - start_time
    print(f"Total time spent: {int(diff_tot // 3600)}:{int((diff_tot % 3600) // 60)}:{(diff_tot % 60):.2f}")


    save_name_1 = f"sac_push_modelA_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    save_name_2 = f"sac_push_modelB_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    save_name_3 = f"sac_push_modelC_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model_a.save(save_name_1)
    model_b.save(save_name_2)
    model_c.save(save_name_3)

    print("\nEvaluation SAC model A.")
    evaluate(model_path=f"{save_name_1}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render,
             env_type=args.env_type, algo_class=SAC)
    print("\nEvaluation SAC model B.")
    evaluate(model_path=f"{save_name_2}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render,
             env_type=args.env_type, algo_class=SAC)
    print("\nEvaluation SAC model C.")
    evaluate(model_path=f"{save_name_3}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render,
             env_type=args.env_type, algo_class=SAC)


if __name__ == "__main__":
    main()