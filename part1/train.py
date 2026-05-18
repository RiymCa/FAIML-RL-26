"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import os
os.environ["WANDB_MODE"] = "offline"
import random
import gymnasium as gym
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt



from agent import Agent, Policy
SEED = 42

os.makedirs("plots", exist_ok=True)
all_configs = {}

def main():

    # setpup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on device: ", device)

    n_episodes = 60000
    algorithms = ['Actor-Critic', 'REINFORCE']
    run_id = 0 

    for alg in algorithms:  

        flag_baseline = [True, False] if alg == 'REINFORCE' else [False]
        
        for baseline in flag_baseline: 

            # set seeds to ensure repr.
            current_seed = SEED + run_id
            run_id += 1
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            random.seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)    

            # print configuration
            print(f"\n\nTraining algorithm {alg} \n\tBaseline {baseline}\n")
                
            # setting up wandb config 
            wandb.init(
                project="FAIML-RL-26-finalV", 
                name=f"{alg}_Baseline_{baseline}", 
                config={"algorithm": alg,
                        "baseline": baseline,
                        "n_episodes": n_episodes,
                        })


            env = gym.make('Hopper-v4')
                            
            print('State space:', env.observation_space)  # state-space
            print('Action space:', env.action_space)  # action-space
                
            dim_state_space = env.observation_space.shape[0]
            dim_action_space = env.action_space.shape[0]

            # agent and policy initialization
            policy = Policy(dim_state_space, dim_action_space).to(device)
            agent = Agent(policy, device=device)

            n_steps_tot = 0
            ep_rewards_list = []

            for ep in range(n_episodes):  
                done = False
                state, info = env.reset(seed=current_seed + ep)  # Reset environment to initial state
                ep_reward = 0.0
                n_steps_inside_episode = 0
                    

                while not done:  # Until the episode is over
                        
                    action, action_log_probs  = agent.get_action(state)  # Sample random action
                    action_numpy = action.cpu().detach().numpy()

                    next_state, reward, terminated, truncated, _ = env.step(action_numpy)  # Step the simulator to the next timestep
                    done = terminated or truncated

                    agent.store_outcome(state, next_state, action_log_probs, reward, done)  

                    state = next_state
                    ep_reward += reward
                    n_steps_tot += 1
                    n_steps_inside_episode += 1

                if alg == "REINFORCE": # reinforce algorithm, just one loss
                    loss, _ = agent.update_policy(baseline=baseline, algorithm=alg)  

                    # log all results on wandb
                    wandb.log({
                        "Episode number": ep,
                        "Episode reward": ep_reward,
                        "n_steps": n_steps_tot,
                        "n_steps_inside_episode": n_steps_inside_episode,
                        "loss": loss,})
                        
                else: # actor ctitic algorithm, actor loss and critic loss
                    actor_loss, critic_loss = agent.update_policy(baseline=baseline, algorithm=alg)  

                    # log all results on wandb
                    wandb.log({
                        "Episode number": ep,
                        "Episode reward": ep_reward,
                        "n_steps": n_steps_tot,
                        "n_steps_inside_episode": n_steps_inside_episode,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,})

                ep_rewards_list.append(ep_reward)

                if ep % 500 == 0:
                    print(f"Episode number: {ep}, Reward: {ep_reward}")
                    
            # rendering of the hopper   
            filename = f"hopper_{alg}_baseline_{baseline}_brain.pth"
            torch.save(agent.policy.state_dict(), filename)
            print(f"Saved trained weights to {filename}")

            env.close()
            wandb.finish()

            # PLOTTING PART
            window = 500
            rewards_arr = np.array(ep_rewards_list)
            rolling_mean = np.convolve(rewards_arr, np.ones(window) / window, mode='valid')

            fig, ax = plt.subplots()
            ax.plot(rewards_arr, alpha=0.3, color='steelblue', label='Episode reward')
            ax.plot(range(window - 1, len(rewards_arr)),
                    rolling_mean, 
                    color='steelblue', 
                    label=f'Moving Mean (w={window})')
            ax.set_xlabel('Episode number')
            ax.set_ylabel('Reward')
            if alg == 'REINFORCE':
                ax.set_title(f'{alg} — Baseline: {baseline}')
            else:
                ax.set_title(f'{alg}')
            ax.legend()
            ax.grid(alpha=0.3)
            fig.savefig(f'plots/{alg}_baseline_{baseline}.pdf')
            plt.close(fig)
            all_configs[f'{alg}_bl{baseline}'] = rewards_arr

    # plot finale con confronto tutte configurazioni
    colors = ['steelblue', 'darkorange', 'green']
    fig, ax = plt.subplots()
    for i, (name, rewards_arr) in enumerate(all_configs.items()):
        rolling_mean = np.convolve(rewards_arr, np.ones(500) / 500, mode='valid')
        ax.plot(range(499, len(rewards_arr)), rolling_mean,
                color=colors[i % len(colors)], label=name)
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Reward moving mean')
    ax.set_title('Comparison of all configurations')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig('plots/comparison.pdf')
    plt.close(fig)
    print("\n\n\nDone!")

    os.system("pmset sleepnow")

if __name__ == '__main__':
    main()