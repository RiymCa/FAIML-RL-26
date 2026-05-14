"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import random

import gymnasium as gym
import torch
import numpy as np
import wandb

from agent import Agent, Policy
SEED = 42


def main():

    # setpup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on device: ", device)

    n_episodes = 60000
    flag_baseline = [False, True]
    flag_normalized_rewards = [False, True]
    run_id = 0 

    for config in flag_baseline:  # loop through the two baseline settings
        for norm in flag_normalized_rewards:  # loop through the two reward normalization settings
            # print whether we are training w or w/o baseline
            
            current_seed = SEED + run_id
            run_id += 1

            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            random.seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)

            
            if not config:
                print(f"\n\nTraining without baseline and norm: {norm}---------------------------------\n\n")
            else:
                print(f"\n\nTraining with baseline b=20 and norm: {norm} --------------------------------- \n\n")

            # setting up wandb config 
            wandb.init(
                project="FAIML-RL-26", 
                name=f"REINFORCE_mean_bl_{config}_norm_{norm}", 
                config={"algorithm": "REINFORCE",
                        "baseline": config,
                        "normalized_rewards": norm,
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

                loss =agent.update_policy(baseline=config, normalize=norm) 

                wandb.log({
                    "Episode number": ep,
                    "Episode reward": ep_reward,
                    "n_steps": n_steps_tot,
                    "n_steps_inside_episode": n_steps_inside_episode,
                    "loss": loss,})

                if ep % 500 == 0:
                    print(f"Episode {ep} Reward: {ep_reward}")
                
            
            filename = f"hopper_baseline_{config}_norm_{norm}_brain.pth"            
            torch.save(agent.policy.state_dict(), filename)
            print(f"Saved trained weights to {filename}")

            env.close()
            wandb.finish()

if __name__ == '__main__':
    main()