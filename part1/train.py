"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
import numpy as np
import wandb

from agent import Agent, Policy

def main():

    # setpup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on device: ", device)

    n_episodes = 1000
    flag_baseline = [False, True]

    for config in flag_baseline:
        # print whether we are training w or w/o baseline
        if not config:
            print("\n\nTraining without baseline")
        else:
            print("\n\nTraining with baseline b=20")  

        # setting up wandb config 
        wandb.init(
            project="FAIML-RL-26", 
            name=f"REINFORCE_baseline_{config}", 
            config={"algorithm": "REINFORCE",
                    "baseline": config,
                    "n_episodes": n_episodes})


        env = gym.make('Hopper-v4')
                       
        print('State space:', env.observation_space)  # state-space
        print('Action space:', env.action_space)  # action-space
        
        dim_state_space = env.observation_space.shape[0]
        dim_action_space = env.action_space.shape[0]

        # agent and policy initialization
        policy = Policy(dim_state_space, dim_action_space).to(device)
        agent = Agent(policy, device=device)


        for ep in range(n_episodes):  
            done = False
            state, info = env.reset()  # Reset environment to initial state
            ep_reward = 0.0

            while not done:  # Until the episode is over
                

                action, action_log_probs  = agent.get_action(state)  # Sample random action
                action_numpy = action.cpu().detach().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action_numpy)  # Step the simulator to the next timestep
                done = terminated or truncated

                agent.store_outcome(state, next_state, action_log_probs, reward, done)  

                state = next_state
                ep_reward += reward

            agent.update_policy(baseline=config) 

            wandb.log({
                "Episode number": ep,
                "Episode reward": ep_reward})
            
            print(f"Episode {ep} Reward: {ep_reward}")
            
        filename = f"hopper_baseline_{config}_brain.pth"
        torch.save(agent.policy.state_dict(), filename)
        print(f"Saved trained weights to {filename}")

        env.close()
        wandb.finish()

if __name__ == '__main__':
    main()