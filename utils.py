# General
from typing import Dict, List, Tuple, Union
import numpy as np

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# OpenAI gym
import gym
from gym import Space, spaces

# Prisoners dilemma environment
from env import PrisonersDilemmaEnv

def get_comb_actions(config, action_seq):
    M = np.zeros((config['num_episodes'], 4))
    for i in range(action_seq.shape[0]):
        # (D, D)
        if action_seq[i, 0] == 0 and action_seq[i, 1] == 0:
            M[i, 0] = 1
        # (D, C)
        elif action_seq[i, 0] == 0 and action_seq[i, 1] == 1:
            M[i, 1] = 1
        # (C, D)
        elif action_seq[i, 0] == 1 and action_seq[i, 1] == 0:
            M[i, 2] = 1
        # (C, C)
        elif action_seq[i, 0] == 1 and action_seq[i, 1] == 1:
            M[i, 3] = 1

    return M.T


def plot_q_vals(params, config, action_seq, q_traj_one, q_traj_two, input_axs=None):

    # Get params
    eps = params['epsilon']
    gamma = params['gamma']
    alpha = params['alpha']

    # Colors
    one_colors = ['b', 'g',]
    two_colors = ['r', 'orange']

    if input_axs is None:
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    else:
        axs = input_axs
    
    axs[0].set_title(f'ϵ=({eps[0]}, {eps[1]}), γ=({gamma[0]}, {gamma[1]})')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Q-value')

    # Agent one
    for i in range(2):
        agent_labels = [f'$p_1: Q_D$', f'$p_1: Q_C$']

        axs[0].plot(q_traj_one[:, i], one_colors[i],  alpha=.8) 
        axs[0].text(
            q_traj_one.shape[0], 
            q_traj_one[-1, i], 
            agent_labels[i], 
            color=one_colors[i], 
            fontsize=15, 
            weight="bold", 
            va="bottom",
        )

    # Agent two
    for i in range(2):
        agent_labels = [f'$p_2: Q_D$', f'$p_2: Q_C$']

        axs[0].plot(q_traj_two[:, i], two_colors[i], alpha=.8) 
        axs[0].text(
            q_traj_two.shape[0], 
            q_traj_two[-1, i], 
            agent_labels[i], 
            color=two_colors[i], 
            fontsize=15, 
            weight="bold", 
            va="top",
        )
    sns.despine()
    axs[1].set_title('Actions')
    #linewidths=0.15, linecolor='k',
    sns.heatmap(get_comb_actions(config, action_seq), annot=False, cbar=False, cmap=['w', 'darkgreen'], vmin=0, vmax=1, yticklabels=['(D, D)', '(D, C)', '(C, D)', '(C, C)'], xticklabels=[], ax=axs[1]);
    axs[1].set_xlabel('Episode')

    if input_axs is None:
        plt.tight_layout()
        plt.show()

def run_IPD(config, params):
    
    # Make the environment
    game_env = PrisonersDilemmaEnv()
    q_table_one = np.zeros(game_env.action_space.n)
    q_table_two = np.zeros(game_env.action_space.n)

    q_traj_one = np.zeros((config['num_episodes'], game_env.action_space.n))
    q_traj_two = np.zeros((config['num_episodes'], game_env.action_space.n))

    rewards_seq = np.zeros((config['num_episodes'], 2))
    action_seq = np.zeros((config['num_episodes'], 2))

    for episode_i in range(config['num_episodes']):
        
        if config['verbose']:
            print(f'EPISODE: {episode_i}')

        # Player one
        if np.random.random() < params['epsilon'][0]:
            action_play_one = np.array(game_env.action_space.sample())
            if config['verbose']: print(f'play_1 --> explore')
            
        else:
            action_play_one = np.random.choice(
                a=np.argwhere((q_table_one == q_table_one.max())).flatten(),
                size=1,
            )
            if config['verbose']:
                print(f'play_{0} --> exploit')
                print(f'choose from: \n{np.argwhere((q_table_one == q_table_one.max())).flatten()}')
        
        # Player two
        if np.random.random() < params['epsilon'][1]:
            action_play_two = np.array(game_env.action_space.sample())

            if config['verbose']: 
                print(f'play_2 --> explore')
            
        else:
            action_play_two = np.random.choice(
                a=np.argwhere((q_table_two == q_table_two.max())).flatten(),
                size=1,
            )

            if config['verbose']:
                print(f'play_{0} --> exploit')
                print(f'choose from: \n{np.argwhere((q_table_two == q_table_two.max())).flatten()}')
        
        # Take a step 
        action_arr = np.array([action_play_one, action_play_two], dtype='object').flatten()
        _, rewards, _, _, _ = game_env.step(action=action_arr)
        
        # Update Q-values
        q_table_one[action_play_one] = q_table_one[action_play_one] + \
            params['alpha'][0] * (rewards[0] + params['gamma'][0] * np.max(q_table_one) - q_table_one[action_play_one])

        q_table_two[action_play_two] = q_table_two[action_play_two] + \
            params['alpha'][1] * (rewards[1] + params['gamma'][1] * np.max(q_table_two) - q_table_two[action_play_two])

        if config['verbose']:
            print(f'actions: \n {action_arr}')
            print(f'rewards: \n {rewards}')
            print(f'q-table p_1: \n {q_table_one}')
            print(f'q-table p_2: \n {q_table_two}')
            print('--- \n')

        # Store trajectory
        rewards_seq[episode_i, :] = rewards
        action_seq[episode_i, :] = action_arr
        # episode x actions x players
        q_traj_one[episode_i] = q_table_one
        q_traj_two[episode_i] = q_table_two

    return q_traj_one, q_traj_two, rewards_seq, action_seq