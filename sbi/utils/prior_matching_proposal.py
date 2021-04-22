import torch
from torch import nn, Tensor, zeros, ones
import sbi.utils as utils
from typing import Any, Optional, Callable
from torch import optim
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from sbi.types import Shape
from tqdm.auto import tqdm
from sbi.utils import BoxUniform


class PriorMatchingProposal(nn.Module):
    def __init__(
        self,
        posterior: "DirectPosterior",
        prior: Any,
        density_estimator: str,
        num_samples_to_estimate: int = 10_000,
        quantile: float = 0.0,
        log_prob_offset: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(density_estimator, str):
            self._build_neural_net = utils.vi_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        assert isinstance(prior, BoxUniform)

        self._dim_theta = int(prior.sample((1,)).shape[1])
        self._neural_net = self._build_neural_net(self._dim_theta)
        self._posterior = deepcopy(posterior)
        self._posterior.net.eval()
        self._prior = prior
        self._thr = identify_cutoff(
            posterior, num_samples_to_estimate, quantile, log_prob_offset
        )
        self._zscore_net = self._posterior.net._transform._transforms[0]
        _, self._zscore_logabsdet = self._zscore_net.inverse(
            zeros(1, self._dim_theta)
        )
        self._zscore_x_net = self._posterior.net._embedding_net

    def sample(self, sample_shape: Shape = torch.Size()) -> Tensor:
        """
        Sample from the prior matching proposal.

        Args:
            sample_shape: Shape of the samples.

        Returns:
            Tensor: Samples.
        """
        xos = self._posterior.default_x.repeat(sample_shape[0], 1)
        xos = self._zscore_x_net(xos)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            base_samples = self._neural_net.sample(sample_shape[0])
            prior_matching_samples, _ = self._posterior.net._transform.inverse(
                base_samples, context=xos
            )
            # unnorm_samples, _ = self._zscore_net.inverse(prior_matching_samples)
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
        xos = self._zscore_x_net(xos)

        with torch.no_grad():
            self._posterior.net.eval()
            self._neural_net.train()

            # z_theta, z_score_lobabsdet = self._zscore_net(theta)
            noise, logabsdet = self._posterior.net._transform(theta, context=xos)
            vi_log_prob = self._neural_net.log_prob(noise)
            return vi_log_prob + logabsdet

    def train(
        self,
        learning_rate: float = 5e-4,
        max_num_epochs: int = 200,
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

        xos = self._posterior.default_x.repeat(num_elbo_particles, 1)
        self._xos = self._zscore_x_net(xos)

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

        data_, logabsdet = self._posterior.net._transform.inverse(
            variational_samples, context=self._xos
        )
        sample_logprob = self._posterior.log_prob(data_)
        below_thr = sample_logprob < self._thr
        target_density = logabsdet  # + self._prior.log_prob(variational_samples)
        target_density[below_thr] = sample_logprob[below_thr]
        print("Below thr target", target_density[below_thr][:5])
        print("Above thr target", target_density[~below_thr][:5])
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


class PriorRejectionProposal:
    def __init__(
        self,
        posterior,
        prior,
        num_samples_to_estimate: int = 10_000,
        quantile: float = 0.0,
        log_prob_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self._dim_theta = int(prior.sample((1,)).shape[1])
        self._posterior = deepcopy(posterior)
        self._posterior.net.eval()
        self._prior = prior
        self._thr = identify_cutoff(
            posterior, num_samples_to_estimate, quantile, log_prob_offset
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = False,
        max_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        """
        Return samples from the `RestrictedPrior`.

        Samples are obtained by sampling from the prior, evaluating them under the
        trained classifier (`RestrictionEstimator`) and using only those that were
        accepted.

        Args:
            sample_shape: Shape of the returned samples.
            show_progress_bars: Whether or not to show a progressbar during sampling.
            max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.

        Returns:
            Samples from the `RestrictedPrior`.
        """

        num_samples = torch.Size(sample_shape).numel()
        num_sampled_total, num_remaining = 0, num_samples
        accepted, acceptance_rate = [], float("Nan")

        # Progress bar can be skipped.
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        # To cover cases with few samples without leakage:
        sampling_batch_size = min(num_samples, max_sampling_batch_size)
        while num_remaining > 0:
            # Sample and reject.
            candidates = self._prior.sample((sampling_batch_size,)).reshape(
                sampling_batch_size, -1
            )
            are_accepted_by_classifier = self.log_prob(candidates)
            samples = candidates[are_accepted_by_classifier.bool()]
            accepted.append(samples)

            # Update.
            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

            # To avoid endless sampling when leakage is high, we raise a warning if the
            # acceptance rate is too low after the first 1_000 samples.
            acceptance_rate = (num_samples - num_remaining) / num_sampled_total

            # For remaining iterations (leakage or many samples) continue sampling with
            # fixed batch size.
            sampling_batch_size = max_sampling_batch_size

        pbar.close()
        print(
            f"The classifier rejected {(1.0 - acceptance_rate) * 100:.1f}% of all "
            f"samples. You will get a speed-up of "
            f"{(1.0 / acceptance_rate - 1.0) * 100:.1f}%.",
        )

        # When in case of leakage a batch size was used there could be too many samples.
        samples = torch.cat(accepted)[:num_samples]
        assert (
            samples.shape[0] == num_samples
        ), "Number of accepted samples must match required samples."

        return samples

    def log_prob(self, theta: Tensor) -> Tensor:
        r"""
        Return whether the parameter lies outside or inside of the support.

        Args:
            theta: Parameters whose label to predict.

        Returns:
            Integers that indicate whether the parameter set is outside (=0) or inside
            (=1).
        """

        log_probs = self._posterior.log_prob(theta)
        predictions = log_probs > self._thr
        return predictions.int()


def identify_cutoff(
    posterior,
    num_samples_to_estimate: int = 10_000,
    quantile: float = 0.0,
    log_prob_offset: float = 0.0,
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
    posterior.net.eval()
    samples = posterior.sample((num_samples_to_estimate,))
    sample_probs = posterior.log_prob(samples)
    # _, logabsdets = posterior.net._transform._transforms[0](samples)
    # sample_probs += logabsdets
    sorted_probs, _ = torch.sort(sample_probs)
    return sorted_probs[int(quantile * num_samples_to_estimate)] + log_prob_offset
