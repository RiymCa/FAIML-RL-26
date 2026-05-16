import argparse
import gymnasium as gym
import time
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rand_wrapper import RandomizationWrapper


def evaluate_success_rate(model, env, n_episodes=50):
    successes = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("is_success", False):
                episode_success = True

        if episode_success:
            successes += 1

    success_rate = (successes / n_episodes) * 100
    return success_rate

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
        default=500000,
        choices=[10000, 50000, 100000, 200000, 500000, 1000000],
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
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
    n_eval_episodes = 100

    print(f"Inizio addestramento Modello A per {timesteps} step")
    model_A = SAC("MultiInputPolicy", env, device="cuda", verbose=0)
    print(f"Device in uso: {model_A.device}")
    model_A.learn(total_timesteps=timesteps)

    print("Valutazione Modello A in corso")
    mean_reward_A, std_reward_A = evaluate_policy(model_A, eval_env, n_eval_episodes=n_eval_episodes)
    success_rate_A = evaluate_success_rate(model_A, eval_env, n_episodes=n_eval_episodes)
    print(f"Risultato Modello A\nPunteggio: {mean_reward_A:.2f}\nTasso di Successo: {success_rate_A:.1f}%")

    end_time_1 = time.time()
    diff1 = end_time_1 - start_time
    print(f"Tempo impiegato A: {int(diff1 // 3600)}:{int((diff1 % 3600) // 60)}:{(diff1 % 60):.2f}")

    print(f"\nInizio addestramento Modello B per {timesteps} step")
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
    print(f"Device in uso: {model_B.device}")
    model_B.learn(total_timesteps=timesteps)

    print("Valutazione Modello B in corso")
    mean_reward_B, std_reward_B = evaluate_policy(model_B, eval_env, n_eval_episodes=n_eval_episodes)
    success_rate_B = evaluate_success_rate(model_B, eval_env, n_episodes=n_eval_episodes)
    print(f"Risultato Modello B\nPunteggio: {mean_reward_B:.2f}\nTasso di Successo: {success_rate_B:.1f}%")

    end_time_2 = time.time()
    diff2 = end_time_2 - end_time_1
    print(f"Tempo impiegato A: {int(diff2 // 3600)}:{int((diff2 % 3600) // 60)}:{(diff2 % 60):.2f}")

    print(f"\nInizio addestramento Modello C per {timesteps} step")
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
    print(f"Device in uso: {model_C.device}")
    model_C.learn(total_timesteps=timesteps)

    print("Valutazione Modello C in corso")
    mean_reward_C, std_reward_C = evaluate_policy(model_C, eval_env, n_eval_episodes=n_eval_episodes)
    success_rate_C = evaluate_success_rate(model_C, eval_env, n_episodes=n_eval_episodes)
    print(f"Risultato Modello\nPunteggio: {mean_reward_C:.2f}\nTasso di Successo: {success_rate_C:.1f}%")

    end_time_3 = time.time()
    diff3 = end_time_3 - end_time_2
    print(f"Tempo impiegato A: {int(diff3 // 3600)}:{int((diff3 % 3600) // 60)}:{(diff3 % 60):.2f}")

    print(f"Modello A: Reward {mean_reward_A:.2f} | Successo: {success_rate_A:.1f}%")
    print(f"Modello B: Reward {mean_reward_B:.2f} | Successo: {success_rate_B:.1f}%")
    print(f"Modello C: Reward {mean_reward_C:.2f} | Successo: {success_rate_C:.1f}%")

    diff_tot = end_time_3 - start_time
    print(f"Tempo impiegato A: {int(diff_tot // 3600)}:{int((diff_tot % 3600) // 60)}:{(diff_tot % 60):.2f}")

if __name__ == "__main__":
    main()