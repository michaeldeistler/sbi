import torch
from torch import nn, Tensor, zeros, ones
import sbi.utils as utils
from typing import Any, Optional, Callable
from torch import optim
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


class PriorMatchingProposal(nn.Module):
    def __init__(
        self,
        posterior: "DirectPosterior",
        prior: Any,
        density_estimator: str,
        num_samples_to_estimate: int = 10_000,
        quantile: float = 0.0,
    ) -> None:
        super().__init__()

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.vi_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        self._dim_theta = int(prior.sample((1,)).shape[1])
        self._neural_net = self._build_neural_net(self._dim_theta)
        self._posterior = deepcopy(posterior)
        self._posterior.net.eval()
        self._prior = prior
        self._thr = self._identify_cutoff(num_samples_to_estimate, quantile)

    def sample(self, sample_shape: torch.Size) -> Tensor:
        self._xos = self._posterior.default_x.repeat(sample_shape[0], 1)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            base_samples = self._neural_net.sample(sample_shape[0])
            prior_matching_samples, _ = self._posterior.net._transform.inverse(
                base_samples, context=self._xos
            )
            return prior_matching_samples

    def log_prob(self, theta: Tensor) -> Tensor:
        self._xos = self._posterior.default_x.repeat(theta.shape[0], 1)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            noise, logabsdet = self._posterior.net._transform(theta, context=self._xos)
            vi_log_prob = self._neural_net.log_prob(noise)
            return vi_log_prob + logabsdet

    def train(
        self,
        learning_rate: float = 5e-4,
        max_num_epochs: int = 1_000,
        num_elbo_particles: int = 100,
        clip_max_norm: Optional[float] = 5.0,
    ) -> nn.Module:
        r"""
        Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            learning_rate: Learning rate for Adam optimizer.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # Move entire net to device for training.
        # self._neural_net.to(self._device)
        self._xos = self._posterior.default_x.repeat(num_elbo_particles, 1)

        self.optimizer = optim.Adam(
            list(self._neural_net.parameters()),
            lr=learning_rate,
        )
        for epoch in range(max_num_epochs):

            # Train for a single epoch.
            self._neural_net.train()
            self.optimizer.zero_grad()
            # Get batches on current device.
            variational_samples = self._neural_net.sample(num_elbo_particles)
            elbo_loss = -torch.mean(self._elbo(variational_samples))

            elbo_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(
                    self._neural_net.parameters(),
                    max_norm=clip_max_norm,
                )
            self.optimizer.step()
            print("Training neural network. Epochs trained: ", epoch, end="\r")

    def _target_density(self, var_samples: Tensor) -> Tensor:

        _, logabsdet = self._posterior.net._transform.inverse(
            var_samples, context=self._xos
        )
        sample_logprob = (
            self._posterior.net._distribution.log_prob(var_samples) - logabsdet
        )
        below_thr = sample_logprob < self._thr
        target_density = logabsdet
        target_density[below_thr] = sample_logprob[below_thr]
        return target_density

    def _elbo(self, variational_samples: Tensor) -> Tensor:
        entropy = -self._neural_net.log_prob(variational_samples)
        mismatch = self._target_density(variational_samples)
        return entropy + mismatch

    def _identify_cutoff(
        self, num_samples_to_estimate: int = 10_000, quantile: float = 0.0
    ) -> Tensor:
        self._posterior.net.eval()
        samples = self._posterior.sample((num_samples_to_estimate,))
        sample_probs = self._posterior.log_prob(samples)
        sorted_probs, _ = torch.sort(sample_probs)
        return sorted_probs[int(quantile * num_samples_to_estimate)]  # - 15.0
