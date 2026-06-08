import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cas4160.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        assert obs.ndim == 1

        obs = ptu.from_numpy(obs)
        action = self.forward(obs.unsqueeze(0)).sample().squeeze(0)
        action = ptu.to_numpy(action)
        # assert action.ndim == 0, action.shape

        return action

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        assert obs.ndim == 2
        if self.discrete:
            logits = self.logits_net(obs)
            assert logits.ndim == 2
            dist = distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            assert mean.ndim == 2
            logstd = self.logstd
            dist = distributions.Normal(loc=mean, scale=logstd.exp())

        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        log_prob = self.forward(obs).log_prob(actions)
        if actions.ndim == 2:
            log_prob = log_prob.sum(axis=-1)
        assert log_prob.ndim == 1, log_prob.shape

        samples_loss = -log_prob * advantages
        assert samples_loss.ndim == 1

        loss = samples_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange=0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # HINT: use torch.clamp to clip values.
        logp = self.forward(obs).log_prob(actions)
        if actions.ndim == 2:
            logp = logp.sum(axis=-1)
        assert logp.ndim == 1, logp.shape
        ratio = torch.exp(logp - old_logp)
        samples_loss = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange) * advantages,
        )
        assert samples_loss.ndim == 1

        loss = samples_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        approx_kl = (old_logp - logp).mean().item()

        return {"PPO Loss": ptu.to_numpy(loss), "kl": approx_kl}
