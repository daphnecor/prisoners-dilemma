# -*- coding: utf-8 -*-
""" Prisoner's dilemma environment with OpenAI gym """

from typing import Dict, List, Tuple, Union

import gym
import numpy as np
from gym import Space, spaces

ActType = Tuple[int, int]
ObsType = None
RenderFrame = None
RewType = Tuple[float, float]


class PrisonersDilemmaEnv(gym.Env):
    """The standard prisoners dilemma.

    Players simultaneously choose whether to defect (0) or cooperate (1).
    """

    action_space: Space[ActType]
    observation_space: Space[ObsType]
    reward_range: Tuple[float, float]

    def __init__(
        self,
        reward_payoff: int,
        temp_payoff: int,
        sucker_payoff: int,
        punish_payoff: int,
    ):
        self.observation_space = spaces.Space()
        self.action_space = spaces.Discrete(2)
        self.reward_range = (sucker_payoff, temp_payoff)
        self.payoff_matrix = self.create_payoff_matrix(
            reward_payoff, temp_payoff, sucker_payoff, punish_payoff
        )

    def create_payoff_matrix(
        self,
        reward_payoff: int,
        temp_payoff: int,
        sucker_payoff: int,
        punish_payoff: int,
    ) -> Dict:
        """Create a payoff matrix.

        Args:
            reward_payoff (int): reward for mutual cooperation
            temp_payoff (int): temptation to defect
            sucker_payoff (int): when exploited by the other player
            punish_payoff (int): punishment for mutual defection

        Returns:
            Dict: payoff matrix
        """
        payoff_dict = {
            "(D, D)": np.array([punish_payoff, punish_payoff]),
            "(D, C)": np.array([temp_payoff, sucker_payoff]),
            "(C, D)": np.array([sucker_payoff, temp_payoff]),
            "(C, C)": np.array([reward_payoff, reward_payoff]),
        }
        return payoff_dict

    def _get_reward(
        self, action_agent_one: int, action_agent_two: int
    ) -> RewType:
        """Return rewards for both agents.

        Args:
            action_agent_one (int): player one's action
            action_agent_two (int): player two's action

        Returns:
            RewType: reward for each player
        """
        if action_agent_one == 0 and action_agent_two == 0:
            return self.payoff_matrix["(D, D)"]

        if action_agent_one == 0 and action_agent_two == 1:
            return self.payoff_matrix["(D, C)"]

        if action_agent_one == 1 and action_agent_two == 0:
            return self.payoff_matrix["(C, D)"]

        if action_agent_one == 1 and action_agent_two == 1:
            return self.payoff_matrix["(C, C)"]

        raise NotImplementedError("Actions not known.")

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, RewType, bool, bool, Dict]:
        """Interact with the environment.

        Args:
            action (ActType): actions taken by the two players.

        Returns:
            Tuple[ObsType, RewType, bool, bool, Dict]: obs, reward, done, _, _
        """

        # Get rewards
        rewards = self._get_reward(*action)

        return (
            None,  # observation
            rewards,  # reward
            True,  # done
            False,  # truncated
            {},  # info
        )

    # Methods below are not used
    def close(self) -> None:  # pylint: disable=missing-function-docstring
        pass

    def render(  # pylint: disable=missing-function-docstring
        self,
    ) -> Union[None, Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(  # pylint: disable=missing-function-docstring
        self,
        *,
        seed: Union[None, int] = None,
        options: Union[None, Dict] = None,
    ) -> Tuple[ObsType, Dict]:
        pass
