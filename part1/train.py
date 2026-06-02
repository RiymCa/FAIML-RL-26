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
import time


from agent import Agent, Policy
SEED = 43

all_configs = {}
time_tracking = {}

def main():

    # setpup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on device: ", device)

    n_episodes = 50000
    algorithms = ['REINFORCE', 'Actor-Critic']
    run_id = 0 

    for alg in algorithms:  

        flag_baseline = [False, True] if alg == 'REINFORCE' else [False]
        
        for baseline in flag_baseline: 

            # set seeds to ensure reprod.
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
            
            start_time = time.time()
            
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

                    # updates 
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

            end_time = time.time()
            tot_duration = end_time - start_time

            print(f"Total duration of the current run: {tot_duration:.4f} seconds ({tot_duration/60:.4f} minutes)")
            time_tracking[f'{alg}_bl_{baseline}'] = tot_duration / 60.0
            env.close()
            wandb.finish()


            all_configs[f'{alg}_bl_{baseline}'] = np.array(ep_rewards_list)

    print("\n\n\nDone!")

    print("\n\n Final results of last 5k episodes")
    print(f"{'Configuration':<35} {'Mean':>15} {'Stddev':>8} {'Max':>8} {'Time(m)':>10}")
    for name, arr in all_configs.items():
        last = arr[-5000:]
        mins = time_tracking[name]
        print(f"{name:<35} {np.mean(last):>15.1f} {np.std(last):>8.1f} {np.max(arr):>8.1f} {mins:>10.1f}")
    os.system("pmset sleepnow")

if __name__ == '__main__':
    main()