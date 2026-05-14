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
    algorithms = ['REINFORCE', 'Actor-Critic']
    run_id = 0 

    for alg in algorithms:  

        flag_baseline = [False, True] if alg == 'REINFORCE' else [False]
        flag_normalized_rewards = [False, True] if alg == 'REINFORCE' else [False]

        for baseline in flag_baseline: 

            for norm in flag_normalized_rewards:  

                # set seeds to ensure repr.
                current_seed = SEED + run_id
                run_id += 1
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                random.seed(current_seed)
                torch.cuda.manual_seed_all(current_seed)

                # print configuration
                print(f"\n\nTraining algorithm {alg} \n\tBaseline {baseline}\n\tnorm: {norm}\n")
                
                # setting up wandb config 
                wandb.init(
                    project="FAIML-RL-26", 
                    name=f"{alg}__Baseline_{baseline}__Norm_{norm}", 
                    config={"algorithm": alg,
                            "baseline": baseline,
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

                    if alg == "REINFORCE":
                        loss, _ = agent.update_policy(baseline=baseline, normalize=norm, 
                                            algorithm=alg)  

                        wandb.log({
                            "Episode number": ep,
                            "Episode reward": ep_reward,
                            "n_steps": n_steps_tot,
                            "n_steps_inside_episode": n_steps_inside_episode,
                            "loss": loss,})
                        
                    else:
                        actor_loss, critic_loss = agent.update_policy(baseline=baseline, normalize=norm, 
                                            algorithm=alg)  

                        wandb.log({
                            "Episode number": ep,
                            "Episode reward": ep_reward,
                            "n_steps": n_steps_tot,
                            "n_steps_inside_episode": n_steps_inside_episode,
                            "actor_loss": actor_loss,
                            "critic_loss": critic_loss,})

                    if ep % 500 == 0:
                        print(f"Episode {ep} Reward: {ep_reward}")
                    
                
                filename = f"hopper{alg}_baseline_{baseline}_norm_{norm}_brain.pth"            
                torch.save(agent.policy.state_dict(), filename)
                print(f"Saved trained weights to {filename}")

                env.close()
                wandb.finish()

if __name__ == '__main__':
    main()