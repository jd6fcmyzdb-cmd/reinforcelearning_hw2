from typing import Optional, Sequence
import numpy as np
import torch

from cas4160.networks.policies import MLPPolicyPG
from cas4160.networks.critics import ValueCritic
from cas4160.networks.reward_predictor import RewardPredictor
from cas4160.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        use_ppo: bool = False,
        n_ppo_epochs: int = 4,
        n_ppo_minibatches: int = 4,
        ppo_cliprange: float = 0.2,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # TODO: initialize reward predictor
        # what should be the input size?
        self.reward_predictor = RewardPredictor(
            input_size=None,
            n_layers=2,
            layer_size=64,
            learning_rate=1e-4
        )

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.use_ppo = use_ppo
        self.n_ppo_epochs = n_ppo_epochs if self.use_ppo else None
        self.n_ppo_minibatches = n_ppo_minibatches if use_ppo else None
        self.ppo_cliprange = ppo_cliprange if use_ppo else None

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
        # truncateds: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        assert all(ob.ndim == 2 for ob in obs)
        assert all(reward.ndim == 1 for reward in rewards)
        assert all(terminal.ndim == 1 for terminal in terminals)

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        # truncateds = np.concatenate(truncateds, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        assert q_values.ndim == 1

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs,
            rewards,
            q_values,
            terminals,  # , truncateds
        )

        assert advantages.ndim == 1

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        if not self.use_ppo:
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-6
                )

            info: dict = self.actor.update(obs, actions, advantages)

            if self.critic is not None:
                for _ in range(self.baseline_gradient_steps):
                    critic_info: dict = self.critic.update(obs, q_values)
                info.update(critic_info)

        else:
            # this part is for PPO
            logp: np.ndarray = self._calculate_log_probs(obs, actions)
            assert logp.ndim == 1

            n_batch = len(obs)
            inds = np.arange(n_batch)
            for _ in range(self.n_ppo_epochs):
                np.random.shuffle(inds)
                # calculate minibatch size to divide a batch to `n_ppo_minibatches` minibatches
                minibatch_size = (
                    n_batch + (self.n_ppo_minibatches - 1)
                ) // self.n_ppo_minibatches
                for start in range(0, n_batch, minibatch_size):
                    end = start + minibatch_size
                    obs_slice, actions_slice, advantages_slice, logp_slice = (
                        arr[inds[start:end]] for arr in (obs, actions, advantages, logp)
                    )

                    if self.normalize_advantages:
                        advantages_slice = (
                            advantages_slice - advantages_slice.mean()
                        ) / (advantages_slice.std() + 1e-6)

                    info: dict = self.actor.ppo_update(
                        obs_slice,
                        actions_slice,
                        advantages_slice,
                        logp_slice,
                        self.ppo_cliprange,
                    )

            assert self.critic is not None
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, q_values)
            info.update(critic_info)
        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        assert all(reward.ndim == 1 for reward in rewards)

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}

            discounted_return = [self._discounted_return(reward) for reward in rewards]
            q_values = discounted_return
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            discounted_return = [
                self._discounted_reward_to_go(reward) for reward in rewards
            ]
            q_values = discounted_return

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values."""
        assert obs.ndim == 2
        if self.critic is None:
            advantages = q_values
        else:
            values = self.critic(ptu.from_numpy(obs)).squeeze(-1)
            values = ptu.to_numpy(values)
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                advantages = q_values - values
                # pass
            else:
                assert rewards.shape == q_values.shape == terminals.shape
                batch_size = obs.shape[0]
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                deltas = (
                    rewards + self.gamma * values[1:] * (1 - terminals) - values[:-1]
                )

                for i in reversed(range(batch_size)):
                    advantages[i] = (
                        self.gamma
                        * self.gae_lambda
                        * advantages[i + 1]
                        * (1 - terminals[i])
                        + deltas[i]
                    )

                advantages = advantages[:-1]

        return advantages

    def _discounted_return(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_return(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 5.9203, 5.9203])
        ```
        """
        assert rewards.ndim == 1

        running_sum = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            running_sum = running_sum * self.gamma + rewards[i]

        res = np.full_like(rewards, running_sum)

        assert len(rewards) == len(res)
        return res

    def _discounted_reward_to_go(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_reward_to_go(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 4.97, 3.])
        ```
        """
        assert rewards.ndim == 1

        res = np.zeros_like(rewards)
        res[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            res[i] = res[i + 1] * self.gamma + rewards[i]

        assert len(rewards) == len(res)
        return res

    def _calculate_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ):
        assert obs.ndim == 2
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        logp = self.actor(obs).log_prob(actions)
        logp = ptu.to_numpy(logp)
        if logp.ndim == 2:
            assert logp.shape[-1] > 1
            logp = np.sum(logp, axis=-1)
        assert logp.ndim == 1, logp.shape
        return logp


if __name__ == "__main__":
    agent = PGAgent(
        ob_dim=2,
        ac_dim=2,
        discrete=True,
        n_layers=2,
        layer_size=64,
        gamma=0.99,
        learning_rate=0.001,
        use_baseline=True,
        use_reward_to_go=True,
        baseline_learning_rate=0.001,
        baseline_gradient_steps=5,
        gae_lambda=0.95,
        normalize_advantages=True,
        use_ppo=True,
        n_ppo_epochs=4,
        n_ppo_minibatches=4,
        ppo_cliprange=0.2,
    )

    # test the discounted return
    rewards = np.array([1.0, 2.0, 3.0])
    total_discounted_return = agent._discounted_return(rewards)
    print(total_discounted_return)

    # test the discounted reward to go
    rewards = np.array([1.0, 2.0, 3.0])
    total_discounted_reward_to_go = agent._discounted_reward_to_go(rewards)
    print(total_discounted_reward_to_go)

    # test the advantage estimation
    obs = np.array(
        [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]]
    )
    rewards = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    q_values = np.array([2, 3, 4, 5, 6, 7])
    q_values_2 = np.array([-2, 3, -4, 5, 6, 7])
    agent.critic.forward = lambda x: torch.tensor([[1], [2], [3], [4], [5], [6]])

    print(torch.tensor([[[1], [2], [3], [4]]]).shape)

    terminals = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    print(obs.shape, rewards.shape, q_values.shape, terminals.shape)
    advantages = agent._estimate_advantage(obs, rewards, q_values, terminals)
    advantages_2 = agent._estimate_advantage(obs, rewards, q_values_2, terminals)

    print(advantages)
    print(advantages_2)

    # # test the log probs
    # obs = np.array([[1.0, 2.0], [3.0, 4.0]])
    # actions = np.array([[1.0, 2.0], [3.0, 4.0]])
    # logp = agent._calculate_log_probs(obs, actions)
    # print(logp)
