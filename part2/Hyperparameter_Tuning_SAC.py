import argparse
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rand_wrapper import RandomizationWrapper


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
        "--timesteps",
        type=int,
        default=100000,
        choices=[10000, 50000, 100000, 200000, 500000, 1000000],
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )
    env = RandomizationWrapper(env, mode=args.sampling_strategy)

    eval_env = gym.make("PandaPush-v3", render_mode="rgb_array", type=args.env_type, reward_type="dense")
    eval_env = Monitor(eval_env)
    eval_env = RandomizationWrapper(eval_env, mode=args.sampling_strategy)

    timesteps = args.timesteps
    n_eval_episodes = 50
    
    print(f"\n--- Inizio addestramento Modello A (SAC Default) per {timesteps} step ---")
    model_A = SAC("MultiInputPolicy", env, device="cuda", verbose=0)
    print(model_A.device)
    model_A.learn(total_timesteps=timesteps)

    print("Valutazione Modello A in corso...")
    mean_reward_A, std_reward_A = evaluate_policy(model_A, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Risultato Modello A: {mean_reward_A:.2f} +/- {std_reward_A:.2f}")

    print(f"\n--- Inizio addestramento Modello B (SAC Bilanciato) per {timesteps} step ---")
    model_B = SAC(
        "MultiInputPolicy",
        env,
        device="cuda",
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=0
    )
    print(model_B.device)
    model_B.learn(total_timesteps=timesteps)

    print("Valutazione Modello B in corso...")
    mean_reward_B, std_reward_B = evaluate_policy(model_B, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Risultato Modello B: {mean_reward_B:.2f} +/- {std_reward_B:.2f}")

    print(f"\n--- Inizio addestramento Modello C (SAC Profondo) per {timesteps} step ---")
    model_C = SAC(
        "MultiInputPolicy",
        env,
        device="cuda",
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=512,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=2,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=0
    )
    print(model_C.device)
    model_C.learn(total_timesteps=timesteps)

    print("Valutazione Modello C in corso...")
    mean_reward_C, std_reward_C = evaluate_policy(model_C, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Risultato Modello C: {mean_reward_C:.2f} +/- {std_reward_C:.2f}")

    print(f"Modello A (Base)       : {mean_reward_A:.2f}")
    print(f"Modello B (Bilanciato) : {mean_reward_B:.2f}")
    print(f"Modello C (Profondo)   : {mean_reward_C:.2f}")

    risultati = {
        "Modello A (Base)": mean_reward_A,
        "Modello B (Bilanciato)": mean_reward_B,
        "Modello C (Profondo)": mean_reward_C
    }

    vincitore = max(risultati, key=risultati.get)
    print(f"Il vincitore è: {vincitore}")


if __name__ == "__main__":
    main()
