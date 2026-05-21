import argparse
import gymnasium as gym
import time
import panda_gym
from stable_baselines3 import PPO
from rand_wrapper import RandomizationWrapper
from eval_sb3 import evaluate

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

    print(f"Starting PPO Model A training for {args.timesteps} steps.")
    env_a = make_train_env(args.env_type, args.sampling_strategy)
    model_a = PPO("MultiInputPolicy", env_a, device="cuda", seed=42, verbose=0)
    print(f"Device in uso: {model_a.device}.")
    model_a.learn(total_timesteps=args.timesteps)
    env_a.close()

    print(f"\nStarting PPO Model B training for {args.timesteps} steps.")
    env_b = make_train_env(args.env_type, args.sampling_strategy)
    model_b = PPO(
        "MultiInputPolicy",
        env_b,
        device="cuda",
        seed=42,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.01,
        verbose=0
    )
    print(f"Device in use: {model_b.device}.")
    model_b.learn(total_timesteps=args.timesteps)
    env_b.close()

    print(f"\nStarting PPO Model C training for {args.timesteps} steps.")
    env_c = make_train_env(args.env_type, args.sampling_strategy)
    model_c = PPO(
        "MultiInputPolicy",
        env_c,
        device="cuda",
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
    model_c.learn(total_timesteps=args.timesteps)
    env_c.close()

    end_time = time.time()
    tot_sec = end_time - start_time
    hours = int(tot_sec // 3600)
    minutes = int((tot_sec % 3600) // 60)
    seconds = tot_sec % 60
    print(f"\nTotal time spent: {hours:02d}:{minutes:02d}:{seconds:05.2f}.")

    save_name_1 = f"ppo_push_modelA_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    save_name_2 = f"ppo_push_modelB_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    save_name_3 = f"ppo_push_modelC_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model_a.save(save_name_1)
    model_b.save(save_name_2)
    model_c.save(save_name_3)

    print("\nEvaluation PPO model A.")
    evaluate(model_path=f"{save_name_1}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)
    print("\nEvaluation PPO model B.")
    evaluate(model_path=f"{save_name_2}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)
    print("\nEvaluation PPO model C.")
    evaluate(model_path=f"{save_name_3}.zip", n_episodes=100, deterministic=not args.stochastic, render=args.render, env_type=args.env_type, algo_class=PPO)


if __name__ == "__main__":
    main()