# Dependencies
from typing import Dict, List, Tuple, Union
import gym
from gym import Space, spaces
from typing import Tuple
import numpy as np

# Specify types
ActType = Tuple[int, int]
ObsType = None
RenderFrame = None
RewType = Tuple[float, float]

class PrisonersDilemmaEnv(gym.Env):
    '''
    The standard prisoners environment. Two-players simultaneously decide whether
    to defect (0) or cooperate (1). The outcome of each game is a set of rewards, 
    where the reward is a function of the actions taken by both agents.
    '''

    action_space: Space[ActType]
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float]

    def __init__(self):
        self.observation_space = spaces.Space()
        self.action_space = spaces.Discrete(2)
        self.reward_range = (0, 3)

    def _get_reward(self, action_agent_one: int, action_agent_two: int) -> RewType:
        
        # (Defect, Defect)
        if action_agent_one == 0 and action_agent_two == 0:
            return np.array([1, 1])
        
        # (Defect, Cooperate)
        elif action_agent_one == 0 and action_agent_two == 1:
            return np.array([3, 0])

        # (Cooperate, Defect)
        elif action_agent_one == 1 and action_agent_two == 0:
            return np.array([0, 3])

        # (Cooperate, Cooperate)
        elif action_agent_one == 1 and action_agent_two == 1:
            return np.array([2, 2])

    def step(self, action: ActType) -> Tuple[ObsType, RewType, bool, bool, Dict]:

        # Get rewards
        rewards = self._get_reward(*action)

        return (
            None, # observation
            rewards, # reward
            True, # done
            False, # truncated
            {} # info
        )

    # Methods below are not used
    def close(self) -> None:
        pass

    def render(self) -> Union[None, Union[RenderFrame, List[RenderFrame]]]:
        pass
    
    def reset(self, *, seed: Union[None, int] = None, options: Union[None, Dict] = None) -> Tuple[ObsType, Dict]:
        pass