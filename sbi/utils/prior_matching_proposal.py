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
        """
        Sample from the prior matching proposal.

        Args:
            sample_shape: Shape of the samples.

        Returns:
            Tensor: Samples.
        """
        xos = self._posterior.default_x.repeat(sample_shape[0], 1)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            base_samples = self._neural_net.sample(sample_shape[0])
            prior_matching_samples, _ = self._posterior.net._transform.inverse(
                base_samples, context=xos
            )
            return prior_matching_samples

    def log_prob(self, theta: Tensor) -> Tensor:
        """
        Log-probability of a parameter under the prior matching proposal.

        Args:
            theta: Parameter set.

        Returns:
            Tensor: Log-probability of the prior matching proposal.
        """
        xos = self._posterior.default_x.repeat(theta.shape[0], 1)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            noise, logabsdet = self._posterior.net._transform(theta, context=xos)
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

        self._xos = self._posterior.default_x.repeat(num_elbo_particles, 1)

        self.optimizer = optim.Adam(
            list(self._neural_net.parameters()),
            lr=learning_rate,
        )
        for epoch in range(max_num_epochs):

            # Train for a single epoch.
            self._neural_net.train()
            self.optimizer.zero_grad()
            elbo_loss = -torch.mean(self._elbo(num_elbo_particles))

            elbo_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(
                    self._neural_net.parameters(),
                    max_norm=clip_max_norm,
                )
            self.optimizer.step()
            print("Training neural network. Epochs trained: ", epoch, end="\r")

    def _target_density(self, variational_samples: Tensor) -> Tensor:
        r"""
        Returns $p(\theta|x), \theta ~ q(\theta)$.
        
        In the above equation, $p(\theta|x)# is the thresholded posterior in latent
        space.

        Args:
            variational_samples: Samples from the variational distribution $q(\theta)$

        Returns:
            Tensor: The pdf.
        """

        _, logabsdet = self._posterior.net._transform.inverse(
            variational_samples, context=self._xos
        )
        sample_logprob = (
            self._posterior.net._distribution.log_prob(variational_samples) - logabsdet
        )
        below_thr = sample_logprob < self._thr
        target_density = logabsdet
        target_density[below_thr] = sample_logprob[below_thr]
        return target_density

    def _elbo(self, num_elbo_particles: int) -> Tensor:
        r"""
        Returns the evidence lower bound.

        Args:
            variational_samples: Samples from the variational distribution $q(\theta)$

        Returns:
            Tensor: The ELBO.
        """
        variational_samples = self._neural_net.sample(num_elbo_particles)
        entropy = -self._neural_net.log_prob(variational_samples)
        mismatch = self._target_density(variational_samples)
        return entropy + mismatch

    def _identify_cutoff(
        self, num_samples_to_estimate: int = 10_000, quantile: float = 0.0
    ) -> Tensor:
        """
        Returns the log-probability at which to threshold the posterior.

        Args:
            num_samples_to_estimate: Number of posterior samples used to estimate the 
                threshold.
            quantile: Of the `num_samples_to_estimate`, we take the log-probablity of 
                the `quantile` lowest one as the threshold.

        Returns:
            Tensor: The threshold.
        """
        self._posterior.net.eval()
        samples = self._posterior.sample((num_samples_to_estimate,))
        sample_probs = self._posterior.log_prob(samples)
        sorted_probs, _ = torch.sort(sample_probs)
        return sorted_probs[int(quantile * num_samples_to_estimate)]
